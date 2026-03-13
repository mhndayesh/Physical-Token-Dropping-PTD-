from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class PTDConfig:
    block_size: int = 6
    segment_size: int = 16
    keep_rate: float = 0.3
    keep_rates: Optional[List[float]] = None
    router_rank: int = 16
    router_queries: int = 8
    router_type: str = "mq"
    router_dim: int = 128
    router_heads: int = 2
    router_layers: int = 1
    router_jitter: float = 0.01
    drop_tokens: bool = True
    ste_gating: bool = True
    prefill_only: bool = False
    recent_window_tokens: int = 128
    router_confidence_threshold: float = 0.55
    max_protected_ratio: float = 0.85


class MultiQueryRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        keep_rate: float,
        num_queries: int,
        rank: int,
        jitter: float,
    ) -> None:
        super().__init__()
        self.keep_rate = keep_rate
        self.jitter = jitter
        self.k_proj = nn.Linear(d_model, rank, bias=False)
        self.queries = nn.Parameter(torch.randn(num_queries, rank))

    def score(
        self,
        segment_embeddings: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_seg, _ = segment_embeddings.shape
        k_seg = max(1, int(n_seg * self.keep_rate))
        keys = self.k_proj(segment_embeddings)
        scores = torch.matmul(self.queries.unsqueeze(0), keys.transpose(1, 2)).max(dim=1).values
        if self.training and self.jitter > 0:
            scores = scores + torch.randn_like(scores) * self.jitter
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        _, topk_idx = torch.topk(scores.detach(), k_seg, dim=-1)
        topk_idx, _ = torch.sort(topk_idx, dim=-1)
        return scores, topk_idx


class TransformerRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        keep_rate: float,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        jitter: float,
    ) -> None:
        super().__init__()
        self.keep_rate = keep_rate
        self.jitter = jitter
        self.in_proj = nn.Linear(d_model, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def score(
        self,
        segment_embeddings: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_seg, _ = segment_embeddings.shape
        k_seg = max(1, int(n_seg * self.keep_rate))
        x = self.in_proj(segment_embeddings)
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        scores = self.score_proj(x).squeeze(-1)
        if self.training and self.jitter > 0:
            scores = scores + torch.randn_like(scores) * self.jitter
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        _, topk_idx = torch.topk(scores.detach(), k_seg, dim=-1)
        topk_idx, _ = torch.sort(topk_idx, dim=-1)
        return scores, topk_idx


@dataclass
class PTDLayerCache:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor


class PTDSparseCache:
    def __init__(self, n_blocks: int, block_size: int) -> None:
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_layers = n_blocks * block_size
        self.entries: List[Optional[PTDLayerCache]] = [None for _ in range(self.n_layers)]

    def _entry(self, layer_idx: int) -> Optional[PTDLayerCache]:
        if layer_idx < 0 or layer_idx >= self.n_layers:
            return None
        return self.entries[layer_idx]

    def get_positions(self, layer_idx: int) -> Optional[torch.Tensor]:
        entry = self._entry(layer_idx)
        if entry is None:
            return None
        return entry.positions

    def get_mask(self, layer_idx: int) -> Optional[torch.Tensor]:
        entry = self._entry(layer_idx)
        if entry is None:
            return None
        return entry.mask

    def set_positions(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        entry = self._entry(layer_idx)
        if entry is None:
            return
        self.entries[layer_idx] = PTDLayerCache(
            key=entry.key,
            value=entry.value,
            positions=positions,
            mask=mask,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self._entry(layer_idx)
        if entry is None:
            key = key_states
            value = value_states
            bsz = key_states.size(0)
            k_len = key_states.size(-2)
            positions = torch.arange(k_len, device=key_states.device).unsqueeze(0).expand(bsz, -1)
            mask = torch.ones(bsz, k_len, dtype=torch.bool, device=key_states.device)
        else:
            key = torch.cat([entry.key, key_states], dim=-2)
            value = torch.cat([entry.value, value_states], dim=-2)
            if entry.positions is not None:
                bsz = key_states.size(0)
                k_len = key_states.size(-2)
                append_pos = torch.arange(k_len, device=key_states.device).unsqueeze(0).expand(bsz, -1)
                positions = torch.cat([entry.positions, append_pos], dim=1)
            else:
                positions = None
            if entry.mask is not None:
                append_mask = torch.ones(key_states.size(0), key_states.size(-2), dtype=torch.bool, device=key_states.device)
                mask = torch.cat([entry.mask, append_mask], dim=1)
            else:
                mask = None
        self.entries[layer_idx] = PTDLayerCache(
            key=key,
            value=value,
            positions=positions,
            mask=mask,
        )
        return key, value

    def get_seq_length(self, layer_idx: int = 0) -> int:
        entry = self._entry(layer_idx)
        if entry is None:
            return 0
        return int(entry.key.size(-2))

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for i, entry in enumerate(self.entries):
            if entry is None:
                continue
            self.entries[i] = PTDLayerCache(
                key=entry.key.index_select(0, beam_idx),
                value=entry.value.index_select(0, beam_idx),
                positions=entry.positions.index_select(0, beam_idx),
                mask=entry.mask.index_select(0, beam_idx),
            )

    def next_position(self, bsz: int, device: torch.device) -> torch.Tensor:
        for entry in self.entries:
            if entry is None:
                continue
            valid_pos = entry.positions.masked_fill(~entry.mask, -1)
            nxt = valid_pos.max(dim=1).values + 1
            return nxt.to(device=device, dtype=torch.long)
        return torch.zeros(bsz, dtype=torch.long, device=device)


def _build_additive_causal_mask(
    position_ids: torch.Tensor,
    token_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    pos_i = position_ids.unsqueeze(-1)
    pos_j = position_ids.unsqueeze(-2)
    causal = pos_i >= pos_j
    q_valid = token_mask.unsqueeze(-1)
    k_valid = token_mask.unsqueeze(-2)
    allowed = causal & q_valid & k_valid
    additive = torch.zeros(
        position_ids.size(0),
        1,
        position_ids.size(1),
        position_ids.size(1),
        dtype=dtype,
        device=position_ids.device,
    )
    additive = additive.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return additive


def _build_additive_causal_mask_qk(
    q_pos: torch.Tensor,
    q_mask: torch.Tensor,
    k_pos: torch.Tensor,
    k_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    if k_mask is None:
        k_mask = torch.ones_like(k_pos, dtype=torch.bool, device=k_pos.device)
    pos_i = q_pos.unsqueeze(-1)
    pos_j = k_pos.unsqueeze(-2)
    causal = pos_i >= pos_j
    q_valid = q_mask.unsqueeze(-1)
    k_valid = k_mask.unsqueeze(-2)
    allowed = causal & q_valid & k_valid
    additive = torch.zeros(
        q_pos.size(0),
        1,
        q_pos.size(1),
        k_pos.size(1),
        dtype=dtype,
        device=q_pos.device,
    )
    additive = additive.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return additive


def _segment_pool(
    x_pad: torch.Tensor,
    m_pad: torch.Tensor,
    n_seg: int,
    seg_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, _, hidden = x_pad.shape
    x_seg = x_pad.view(bsz, n_seg, seg_size, hidden)
    m_seg = m_pad.view(bsz, n_seg, seg_size)
    denom = m_seg.sum(dim=2, keepdim=True).clamp_min(1).to(x_pad.dtype)
    pooled = (x_seg * m_seg.unsqueeze(-1).to(x_pad.dtype)).sum(dim=2) / denom
    seg_valid = m_seg.any(dim=2)
    return pooled, seg_valid


def _topk_with_mandatory(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    mandatory_mask: Optional[torch.Tensor],
    keep_rate: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, n_seg = scores.shape
    base_k = max(1, int(n_seg * keep_rate))
    if mandatory_mask is None:
        mandatory_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    mandatory_mask = mandatory_mask & valid_mask
    max_mandatory = int(mandatory_mask.sum(dim=-1).max().item()) if mandatory_mask.numel() > 0 else 0
    k_sel = max(base_k, max_mandatory)
    k_sel = min(max(1, k_sel), n_seg)
    boosted = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    if mandatory_mask.any():
        boosted = boosted + mandatory_mask.to(boosted.dtype) * 1e6
    _, topk_idx = torch.topk(boosted.detach(), k_sel, dim=-1)
    topk_idx, _ = torch.sort(topk_idx, dim=-1)
    sel = torch.zeros(bsz, n_seg, dtype=torch.bool, device=scores.device)
    sel.scatter_(1, topk_idx, True)
    sel = sel & valid_mask
    return topk_idx, sel


class PTDQwen2ForCausalLM(nn.Module):
    def __init__(self, base_model: Qwen2ForCausalLM, ptd: PTDConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.ptd = ptd

        n_layers = base_model.config.num_hidden_layers
        if n_layers % ptd.block_size != 0:
            raise ValueError(
                f"num_hidden_layers ({n_layers}) must be divisible by block_size ({ptd.block_size})"
            )
        n_blocks = n_layers // ptd.block_size
        d_model = base_model.config.hidden_size

        routers = []
        for _ in range(n_blocks):
            if ptd.router_type == "transformer":
                routers.append(
                    TransformerRouter(
                        d_model=d_model,
                        keep_rate=ptd.keep_rate,
                        hidden_dim=ptd.router_dim,
                        num_heads=ptd.router_heads,
                        num_layers=ptd.router_layers,
                        jitter=ptd.router_jitter,
                    )
                )
            else:
                routers.append(
                    MultiQueryRouter(
                        d_model=d_model,
                        keep_rate=ptd.keep_rate,
                        num_queries=ptd.router_queries,
                        rank=ptd.router_rank,
                        jitter=ptd.router_jitter,
                    )
                )
        self.routers = nn.ModuleList(routers)
        ref_param = next(self.base_model.parameters())
        self.routers.to(device=ref_param.device, dtype=ref_param.dtype)
        if ptd.keep_rates:
            self.set_keep_rates(ptd.keep_rates, scale=1.0)
        self.layer_groups = [
            [base_model.model.layers[j] for j in range(i * ptd.block_size, (i + 1) * ptd.block_size)]
            for i in range(n_blocks)
        ]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        ptd_config: Optional[PTDConfig] = None,
        **kwargs: Any,
    ) -> "PTDQwen2ForCausalLM":
        if ptd_config is None:
            ptd_config = PTDConfig()
        base = Qwen2ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(base_model=base, ptd=ptd_config)

    @property
    def config(self):
        return self.base_model.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        return self

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            if name.startswith("routers."):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def router_parameters(self) -> Iterable[nn.Parameter]:
        return self.routers.parameters()

    def set_keep_rate(self, keep_rate: float) -> None:
        self.ptd.keep_rate = keep_rate
        for router in self.routers:
            router.keep_rate = keep_rate

    def set_keep_rates(self, keep_rates: List[float], scale: float = 1.0) -> None:
        if len(keep_rates) != len(self.routers):
            raise ValueError("keep_rates length must match number of routers.")
        self.ptd.keep_rates = list(keep_rates)
        for router, base in zip(self.routers, keep_rates):
            router.keep_rate = max(0.0, min(1.0, base * scale))

    def set_drop_tokens(self, drop_tokens: bool) -> None:
        self.ptd.drop_tokens = drop_tokens

    def set_prefill_only(self, prefill_only: bool) -> None:
        self.ptd.prefill_only = prefill_only

    def set_recent_window(self, recent_window_tokens: int) -> None:
        self.ptd.recent_window_tokens = max(0, int(recent_window_tokens))

    def should_fallback(self, aux: Dict[str, torch.Tensor]) -> bool:
        conf = aux.get("router_confidence")
        if conf is not None and conf.numel() > 0:
            if float(conf.float().mean().item()) < float(self.ptd.router_confidence_threshold):
                return True
        protected_ratio = aux.get("protected_ratio")
        if protected_ratio is not None and protected_ratio.numel() > 0:
            if float(protected_ratio.float().mean().item()) > float(self.ptd.max_protected_ratio):
                return True
        return False

    def init_ptd_cache(self) -> PTDSparseCache:
        return PTDSparseCache(n_blocks=len(self.layer_groups), block_size=self.ptd.block_size)

    def ptd_config_dict(self) -> Dict[str, Any]:
        return asdict(self.ptd)

    def _forward_hidden_with_aux(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_block_hidden: bool = False,
        use_cache: bool = False,
        ptd_cache: Optional["PTDSparseCache"] = None,
        mandatory_keep_mask: Optional[torch.Tensor] = None,
        force_keep_last_n: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        model = self.base_model.model
        if inputs_embeds is None:
            hidden = model.embed_tokens(input_ids)
        else:
            hidden = inputs_embeds

        bsz, seq_len, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        if position_ids is None:
            if use_cache and ptd_cache is not None:
                start = ptd_cache.next_position(bsz=bsz, device=device).unsqueeze(1)
                offset = torch.arange(seq_len, device=device).unsqueeze(0)
                position_ids = start + offset
            else:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        if attention_mask is None:
            token_mask = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
        else:
            token_mask = attention_mask.to(torch.bool)
        if use_cache and ptd_cache is None:
            raise ValueError("use_cache=True requires ptd_cache for PTD sparse caching.")

        if mandatory_keep_mask is None:
            mandatory_keep_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        else:
            mandatory_keep_mask = mandatory_keep_mask.to(device=device, dtype=torch.bool)
        force_keep_last_n = max(force_keep_last_n, int(getattr(self.ptd, "recent_window_tokens", 0)))
        if force_keep_last_n > 0:
            last_tok = token_mask.sum(dim=-1).clamp_min(1)
            for bi in range(bsz):
                end = int(last_tok[bi].item())
                start = max(0, end - force_keep_last_n)
                mandatory_keep_mask[bi, start:end] = True
        mandatory_keep_mask = mandatory_keep_mask & token_mask

        seg_size = self.ptd.segment_size
        selection_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        gate_means = []
        segment_selections = []
        router_entropies = []
        router_confidences = []
        segment_scores = []
        segment_valids = []
        block_hidden = [] if return_block_hidden else None

        for block_idx, (router, layers) in enumerate(zip(self.routers, self.layer_groups)):
            seg_size_cur = max(1, min(seg_size, seq_len)) if use_cache else seg_size
            pad_len = (seg_size_cur - (seq_len % seg_size_cur)) % seg_size_cur
            if pad_len > 0:
                x_pad = F.pad(hidden, (0, 0, 0, pad_len))
                p_pad = F.pad(position_ids, (0, pad_len), value=position_ids[:, -1:].max().item() + 1)
                m_pad = F.pad(token_mask, (0, pad_len), value=False)
                i_pad = F.pad(
                    torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1),
                    (0, pad_len),
                    value=-1,
                )
            else:
                x_pad = hidden
                p_pad = position_ids
                m_pad = token_mask
                i_pad = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

            n_pad = x_pad.size(1)
            n_seg = n_pad // seg_size_cur
            pooled, seg_valid = _segment_pool(x_pad, m_pad, n_seg, seg_size_cur)
            if pad_len > 0:
                mandatory_pad = F.pad(mandatory_keep_mask, (0, pad_len), value=False)
            else:
                mandatory_pad = mandatory_keep_mask
            mandatory_seg = mandatory_pad.view(bsz, n_seg, seg_size_cur).any(dim=-1)
            seg_scores, _ = router.score(pooled, valid_mask=seg_valid)
            seg_ix, seg_selected = _topk_with_mandatory(
                seg_scores,
                seg_valid,
                mandatory_seg,
                keep_rate=router.keep_rate,
            )
            segment_scores.append(seg_scores)
            segment_valids.append(seg_valid)
            segment_selections.append(seg_selected)

            scores_f = seg_scores.float().masked_fill(~seg_valid, -1e9)
            probs = torch.softmax(scores_f, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            router_entropies.append(entropy)
            router_confidences.append((probs * seg_selected.float()).sum(dim=-1).mean())

            tok_ix = (
                seg_ix.unsqueeze(-1) * seg_size_cur + torch.arange(seg_size_cur, device=device)
            ).view(bsz, -1)
            k_tok = tok_ix.size(1)
            bat_ix = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, k_tok)

            if self.ptd.drop_tokens:
                x_sp = x_pad[bat_ix, tok_ix]
                p_sp = p_pad[bat_ix, tok_ix]
                m_sp = m_pad[bat_ix, tok_ix]
                idx_sp = i_pad[bat_ix, tok_ix]

                if self.training and self.ptd.ste_gating:
                    sel_soft = torch.gather(seg_scores, 1, seg_ix)
                    gate_soft = torch.sigmoid(sel_soft)
                    tok_gate = gate_soft.unsqueeze(-1).repeat(1, 1, seg_size_cur).view(bsz, k_tok, 1)
                    ste_gate = (tok_gate - tok_gate.detach()) + 1.0
                    x_sp = x_sp * ste_gate

                attn_sp = None
                if not use_cache:
                    attn_sp = _build_additive_causal_mask(p_sp, m_sp, dtype=dtype)
                cos_sp, sin_sp = model.rotary_emb(x_sp, p_sp)
                for layer_idx, layer in enumerate(layers):
                    layer_global_idx = int(getattr(layer, "layer_idx", block_idx * len(layers) + layer_idx))
                    k_pos = p_sp
                    k_mask = m_sp
                    if use_cache and ptd_cache is not None:
                        past_pos = ptd_cache.get_positions(layer_global_idx)
                        past_mask = ptd_cache.get_mask(layer_global_idx)
                        if past_pos is not None:
                            if past_mask is None:
                                past_mask = torch.ones_like(past_pos, dtype=torch.bool, device=past_pos.device)
                            k_pos = torch.cat([past_pos, p_sp], dim=1)
                            k_mask = torch.cat([past_mask, m_sp], dim=1)
                        attn_sp = _build_additive_causal_mask_qk(
                            q_pos=p_sp,
                            q_mask=m_sp,
                            k_pos=k_pos,
                            k_mask=k_mask,
                            dtype=dtype,
                        )
                    x_sp = layer(
                        hidden_states=x_sp,
                        attention_mask=attn_sp,
                        position_ids=p_sp,
                        position_embeddings=(cos_sp, sin_sp),
                        past_key_values=ptd_cache if use_cache else None,
                        use_cache=use_cache,
                        output_attentions=False,
                    )
                    if isinstance(x_sp, (tuple, list)):
                        x_sp = x_sp[0]
                    if use_cache and ptd_cache is not None:
                        ptd_cache.set_positions(layer_global_idx, k_pos, k_mask)
                x_pad = x_pad.clone()
                x_pad[bat_ix, tok_ix] = x_sp
                hidden = x_pad[:, :seq_len, :]

                valid = (idx_sp >= 0) & (idx_sp < seq_len) & m_sp
                selection_mask[bat_ix[valid], idx_sp[valid]] = True
            else:
                if self.training:
                    soft_all = torch.sigmoid(seg_scores).unsqueeze(-1).repeat(1, 1, seg_size_cur).view(bsz, n_pad, 1)
                    gate_mean = (soft_all.squeeze(-1) * m_pad.to(soft_all.dtype)).sum() / m_pad.to(
                        soft_all.dtype
                    ).sum().clamp_min(1.0)
                    gate_means.append(gate_mean)
                    if self.ptd.ste_gating:
                        gate_full = (soft_all - soft_all.detach()) + 1.0
                    else:
                        gate_full = soft_all
                    x_pad = x_pad * gate_full

                attn_full = None
                if not use_cache:
                    attn_full = _build_additive_causal_mask(p_pad, m_pad, dtype=dtype)
                cos_full, sin_full = model.rotary_emb(x_pad, p_pad)
                for layer_idx, layer in enumerate(layers):
                    layer_global_idx = int(getattr(layer, "layer_idx", block_idx * len(layers) + layer_idx))
                    k_pos = p_pad
                    k_mask = m_pad
                    if use_cache and ptd_cache is not None:
                        past_pos = ptd_cache.get_positions(layer_global_idx)
                        past_mask = ptd_cache.get_mask(layer_global_idx)
                        if past_pos is not None:
                            if past_mask is None:
                                past_mask = torch.ones_like(past_pos, dtype=torch.bool, device=past_pos.device)
                            k_pos = torch.cat([past_pos, p_pad], dim=1)
                            k_mask = torch.cat([past_mask, m_pad], dim=1)
                        attn_full = _build_additive_causal_mask_qk(
                            q_pos=p_pad,
                            q_mask=m_pad,
                            k_pos=k_pos,
                            k_mask=k_mask,
                            dtype=dtype,
                        )
                    x_pad = layer(
                        hidden_states=x_pad,
                        attention_mask=attn_full,
                        position_ids=p_pad,
                        position_embeddings=(cos_full, sin_full),
                        past_key_values=ptd_cache if use_cache else None,
                        use_cache=use_cache,
                        output_attentions=False,
                    )
                    if isinstance(x_pad, (tuple, list)):
                        x_pad = x_pad[0]
                    if use_cache and ptd_cache is not None:
                        ptd_cache.set_positions(layer_global_idx, k_pos, k_mask)
                hidden = x_pad[:, :seq_len, :]

                idx_sp = i_pad[bat_ix, tok_ix]
                m_sp = m_pad[bat_ix, tok_ix]
                valid = (idx_sp >= 0) & (idx_sp < seq_len) & m_sp
                selection_mask[bat_ix[valid], idx_sp[valid]] = True

            if return_block_hidden:
                block_hidden.append(hidden)

        hidden = model.norm(hidden)
        aux = {
            "selection_mask": selection_mask,
            "token_mask": token_mask,
            "gate_means": torch.stack(gate_means) if gate_means else torch.empty(0, device=device, dtype=dtype),
            "segment_selection": (
                torch.stack(segment_selections) if segment_selections else torch.empty(0, device=device, dtype=torch.bool)
            ),
            "segment_scores": (
                torch.stack(segment_scores) if segment_scores else torch.empty(0, device=device, dtype=dtype)
            ),
            "segment_valid": (
                torch.stack(segment_valids) if segment_valids else torch.empty(0, device=device, dtype=torch.bool)
            ),
            "router_entropy": (
                torch.stack(router_entropies)
                if router_entropies
                else torch.empty(0, device=device, dtype=torch.float32)
            ),
            "router_confidence": (
                torch.stack(router_confidences)
                if router_confidences
                else torch.empty(0, device=device, dtype=torch.float32)
            ),
            "mandatory_keep_mask": mandatory_keep_mask,
            "protected_ratio": mandatory_keep_mask.float().sum(dim=-1) / token_mask.float().sum(dim=-1).clamp_min(1.0),
        }
        if return_block_hidden:
            aux["block_hidden"] = block_hidden
        return hidden, aux

    def forward_with_aux(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_block_hidden: bool = False,
        use_cache: bool = False,
        ptd_cache: Optional["PTDSparseCache"] = None,
        mandatory_keep_mask: Optional[torch.Tensor] = None,
        force_keep_last_n: int = 0,
    ) -> Tuple[CausalLMOutputWithPast, Dict[str, torch.Tensor]]:
        hidden, aux = self._forward_hidden_with_aux(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_block_hidden=return_block_hidden,
            use_cache=use_cache,
            ptd_cache=ptd_cache,
            mandatory_keep_mask=mandatory_keep_mask,
            force_keep_last_n=force_keep_last_n,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.base_model.lm_head(hidden[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.base_model.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.base_model.config.vocab_size,
            )
        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=ptd_cache if use_cache else None,
            hidden_states=None,
            attentions=None,
        )
        return out, aux

    @torch.no_grad()
    def generate_prefill_dense(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        mandatory_keep_mask: Optional[torch.Tensor] = None,
        force_keep_last_n: int = 0,
        **generate_kwargs: Any,
    ) -> torch.LongTensor:
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError("generate_prefill_dense currently supports batch size 1 only.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        _, aux = self.forward_with_aux(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mandatory_keep_mask=mandatory_keep_mask,
            force_keep_last_n=force_keep_last_n,
        )
        if self.should_fallback(aux):
            return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        sel = aux["selection_mask"]
        if mandatory_keep_mask is not None:
            sel = sel | mandatory_keep_mask.to(sel.device, dtype=torch.bool)
        if force_keep_last_n > 0:
            end = int(attention_mask[0].sum().item())
            start = max(0, end - force_keep_last_n)
            sel[:, start:end] = True
        compact = input_ids[0][sel[0]]
        compact_mask = torch.ones_like(compact, dtype=attention_mask.dtype)
        return self.base_model.generate(
            input_ids=compact.unsqueeze(0),
            attention_mask=compact_mask.unsqueeze(0),
            **generate_kwargs,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        ptd_cache = kwargs.pop("ptd_cache", None)
        ptd_use_sparse_cache = bool(kwargs.pop("ptd_use_sparse_cache", False))
        mandatory_keep_mask = kwargs.pop("mandatory_keep_mask", None)
        force_keep_last_n = int(kwargs.pop("force_keep_last_n", 0))
        if isinstance(past_key_values, PTDSparseCache):
            ptd_cache = past_key_values
            ptd_use_sparse_cache = True

        if ptd_use_sparse_cache:
            if ptd_cache is None:
                ptd_cache = self.init_ptd_cache()
            out, _ = self.forward_with_aux(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                logits_to_keep=logits_to_keep,
                use_cache=True if use_cache is None else bool(use_cache),
                ptd_cache=ptd_cache,
                mandatory_keep_mask=mandatory_keep_mask,
                force_keep_last_n=force_keep_last_n,
            )
            return out

        # Default cache path falls back to dense HF cache semantics.
        if use_cache or past_key_values is not None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        out, _ = self.forward_with_aux(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            logits_to_keep=logits_to_keep,
            mandatory_keep_mask=mandatory_keep_mask,
            force_keep_last_n=force_keep_last_n,
        )
        return out
