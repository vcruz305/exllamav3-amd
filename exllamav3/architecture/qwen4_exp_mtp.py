from __future__ import annotations
from typing_extensions import override
import os
import torch
from ..util.device_copy import to_device
import weakref

from ..model.config import Config
from ..model.model import Model
from ..modules import Embedding, Linear, GatedResidual
from ..modules.module import Module
from ..modules.quant.exl3 import LinearEXL3
from ..modules.arch_specific.qwen4_exp_mtp import Qwen4ExpMTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qwen4_exp import Qwen4ExpConfig

"""
MTP (multi-token prediction) draft head for Qwen3.8-Flash-Next: input combine over the trunk's
PRE-collapse hyper-connection stream stack (exported by the trunk's final mixer) plus the next
token's embedding, one full qwen4_exp decoder block (QSA attention, MoE, gated-residual sites),
and its own combine-less mixer. Shares the trunk's embedding and lm_head.

No reference implementation exists for this head; the input-combine stream handling
(Qwen4ExpMTPInputLayer.stream_tap) is a semantic guess that must be confirmed by acceptance
rate on the full model.
"""


class Qwen4ExpMTPStackOut(Module):
    """
    Terminal module of the MTP draft chain: passes the decoder block's stream stack through
    FLATTENED (bsz, seq, hc_mult * hidden) instead of collapsing it, so the model's forward
    output can feed the next drafting step's target_hidden (symmetric with the trunk's pre-mixer
    stack export). The mixer is owned here as a submodule (so it loads with the model) and is
    applied by sample_from_state() before the shared lm_head.
    """

    def __init__(self, config, key: str, mixer: GatedResidual):
        super().__init__(config, key, None)
        self.mixer = mixer
        self.register_submodule(mixer)

    def optimizer_targets(self):
        return []

    # The compile step collects a top-level module's output tensors by ITS key prefix; this
    # module's own key ("mtp_stack_out") names no tensors, so it must hand the collection to
    # the owned mixer (prefix "mtp.hyper_connection_mixer."), or the mixer's tensors stay in
    # the qtensors files and never reach the compiled shards
    def get_compile_sizes(self, stc):
        return self.mixer.get_compile_sizes(stc)

    def get_compile_tensors(self, stc):
        return self.mixer.get_compile_tensors(stc)

    def forward(self, x, params, out_dtype = None):
        return x.flatten(-2).half()


class Qwen4ExpMTPModel(Model):

    def __init__(
        self,
        config: Qwen4ExpConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        from .qwen4_exp import build_qwen4_block

        self.input_layer = Qwen4ExpMTPInputLayer(
            config = config,
            key = "mtp",
            hidden_size = config.hidden_size,
            hc_mult = config.hc_mult,
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        )
        self.modules = [self.input_layer]
        self.first_block_idx = len(self.modules)

        for idx in range(config.mtp_num_hidden_layers):
            self.modules.append(
                build_qwen4_block(
                    config,
                    f"mtp.layers.{idx}",
                    idx,
                    "full_attention",
                    qbits_key = "mtp_bits",
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # The draft chain's output is the flattened PRE-mixer stream stack (it feeds the next
        # drafting step's target_hidden); sample_from_state applies the mixer + shared lm_head
        self.stack_out = Qwen4ExpMTPStackOut(
            config,
            "mtp_stack_out",
            GatedResidual(
                config = config,
                key = "mtp.hyper_connection_mixer",
                hc_mult = config.hc_mult,
                hidden_size = config.hidden_size,
                rms_norm_eps = config.rms_norm_eps,
                use_combine = False,
                out_dtype = torch.half,
            ),
        )
        self.modules.append(self.stack_out)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 4,
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")

    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and have the trunk's final mixer
        export the pre-collapse stream stack as the draft input state.
        """
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)

        target_embed = None
        for m in target.modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        target_mixer = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_mixer, GatedResidual) and not target_mixer.use_combine, \
            "Expected the trunk's combine-less mixer immediately before lm_head"
        self.draft_verifier_params.update({
            "export_state_norm_keys": {target_mixer.key},
        })

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, max_chunk_size):
        return {}

    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        # state is the flattened pre-mixer stream stack; collapse it before the shared head
        mixer = self.stack_out.mixer
        bsz, seq, _ = state.shape
        stack = to_device(state, mixer.device).view(bsz, seq, mixer.hc_mult, mixer.hidden_size)
        state = mixer.forward(stack, params)
        ll = self.attached_model().logit_layer_idx
        lm = self.attached_model().modules[ll]
        logits = lm.prepare_for_device(state, params)
        sub = self._draft_head(lm)
        if sub is not None:
            # Drafting scores only the vocab prefix [0, N) plus the special/control block at the
            # top of the vocab; the target verifies every drafted token, so output is unchanged
            # and only the acceptance rate can move
            head, tail, tail_start = sub
            flat = logits.view(-1, logits.shape[-1])
            lo = head.forward(flat, params)
            hi = tail.forward(flat, params)
            V = self.attached_model().config.vocab_size
            full = torch.full((flat.shape[0], V), float("-inf"), dtype = lo.dtype, device = lo.device)
            full[:, :lo.shape[-1]] = lo
            n_tail = min(hi.shape[-1], V - tail_start)
            full[:, tail_start:tail_start + n_tail] = hi[:, :n_tail]
            logits = full.view(bsz, seq, V)
        else:
            logits = lm.forward(logits, params)
        if params.get("export_draft_conf"):
            logits = logits[..., :self.attached_model().config.vocab_size]
            conf, ids = torch.max(logits, dim = -1)
            params["draft_conf"] = conf
            return ids
        return torch.argmax(logits, dim = -1)

    def _draft_head(self, lm):
        """
        (head, tail, tail_start) for a reduced-vocabulary draft head, or None for the full head.

        head: the EXL3 lm_head sliced to output columns [0, N), N = EXL3_MTP_DRAFT_VOCAB (default
        98304, 0 = full head) rounded down to a multiple of 128. BPE merge order puts frequent tokens at low ids: on the
        Flash-Next tokenizer 98,304 of 248,320 columns cover 98.2 % of calibration-text tokens.
        tail: the same head sliced to the special/control block at the top of the vocabulary
        (from the tokenizer's first special id, rounded down to 128, to the end). The drafter
        proposes these often (<|im_start|>, </think>, ...: ~15 % of drafts on chat-less
        prompts), so they must stay draftable.

        Both slices are exact (same trellis tiles, Hadamard and sign vectors as the full head).
        The trellis is tiled [k/16, n/16, 16K], so each slice is one contiguous copy made at
        first use (65,536 columns at 5 bpw: ~105 MB).
        """
        n = _draft_vocab_cols()
        cached = getattr(self, "_draft_head_cache", None)
        if cached is not None and cached[0] is lm.inner and cached[1] == n:
            return cached[2]
        sub = None
        inner = getattr(lm, "inner", None)
        tail_start = _special_start(self.attached_model().config.vocab_size)
        if n and isinstance(inner, LinearEXL3) and n < tail_start and not lm.lora_a_tensors \
                and lm.softcap == 0.0 and lm.pre_scale == 1.0 and lm.post_scale == 1.0:
            def slice_cols(a, b):
                return LinearEXL3(
                    None, inner.in_features, b - a,
                    suh = inner.suh,
                    svh = inner.svh[a:b].contiguous(),
                    trellis = inner.trellis[:, a // 16 : b // 16, :].contiguous(),
                    mcg = inner.mcg_tensor,
                    mul1 = inner.mul1_tensor,
                    bias = inner.bias[a:b].contiguous() if inner.bias is not None else None,
                    out_dtype = inner.out_dtype,
                    key = (inner.key or "lm_head") + f".draft{a}_{b}",
                )
            sub = (slice_cols(0, n), slice_cols(tail_start, inner.out_features), tail_start)
        self._draft_head_cache = (inner, n, sub)
        return sub


def _special_start(vocab_size: int) -> int:
    """First column of the special/control block, rounded down to a multiple of 128. Qwen3.x
    tokenizers put all added tokens above the BPE vocab (Flash-Next: 248044 onward)."""
    s = int(os.environ.get("EXL3_MTP_DRAFT_SPECIAL_START", "248044"))
    return min((s // 128) * 128, vocab_size)


def _draft_vocab_cols() -> int:
    # Default 98304: token-identical greedy output to the full head and +5 % six-prompt decode on
    # gfx1151 (README.strix-halo.md). 0 = full head.
    n = int(os.environ.get("EXL3_MTP_DRAFT_VOCAB", "98304") or 0)
    return (n // 128) * 128 if n > 0 else 0
