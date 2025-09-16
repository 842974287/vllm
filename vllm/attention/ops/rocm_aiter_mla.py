# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op, is_torch_equal_or_newer


def get_aiter_mla_metadata(max_batch_size: int, block_size: int,
                           max_block_per_batch: int,
                           device: torch.device) -> tuple[torch.Tensor, ...]:
    paged_kv_indices = torch.zeros(max_batch_size * max_block_per_batch,
                                   dtype=torch.int32,
                                   device=device)
    paged_kv_indptr = torch.zeros(max_batch_size + 1,
                                  dtype=torch.int32,
                                  device=device)
    paged_kv_last_page_lens = torch.full((max_batch_size, ),
                                         block_size,
                                         dtype=torch.int32)
    qo_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int, device=device)
    return paged_kv_indices, paged_kv_indptr, paged_kv_last_page_lens, qo_indptr


def aiter_mla_decode_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    sm_scale: float,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    logit_cap: float = 0.0,
    work_meta_data: Optional[torch.Tensor] = None,
    work_info_set: Optional[torch.Tensor] = None,
    work_indptr: Optional[torch.Tensor] = None,
    reduce_indptr: Optional[torch.Tensor] = None,
    reduce_final_map: Optional[torch.Tensor] = None,
    reduce_partial_map: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
):

    torch.ops.vllm.rocm_aiter_mla_decode_fwd(q,
                                             kv_buffer.view(
                                                 -1, 1, 1, q.shape[-1]),
                                             o,
                                             qo_indptr,
                                             max_seqlen_qo,
                                             kv_indptr,
                                             kv_indices,
                                             kv_last_page_lens,
                                             sm_scale=sm_scale,
                                             logit_cap=logit_cap,
                                             work_meta_data=work_meta_data,
                                             work_info_set=work_info_set,
                                             work_indptr=work_indptr,
                                             reduce_indptr=reduce_indptr,
                                             reduce_final_map=reduce_final_map,
                                             reduce_partial_map=reduce_partial_map,
                                             q_scale=q_scale,
                                             kv_scale=kv_scale)


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8=False,
    q_scale=None,
    kv_scale=None,
) -> torch.Tensor:

    if is_fp8:
        scale *= q_scale * kv_scale
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)

    m = attn_weights.max(-1).values

    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))

    l = attn_weights_exp.sum(-1)

    if is_fp8:
        attn_weights_fp8 = attn_weights_exp.to(torch.float8_e4m3fnuz)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())

    out = out / l.transpose(0, 1).unsqueeze(-1)

    if is_fp8:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):
    is_fp8 = q.dtype == torch.float8_e4m3fnuz

    if is_fp8:
        q = q.to(torch.float)
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o, lse = ref_masked_attention(
            q,
            k,
            v,
            sm_scale,
            dtype,
            is_causal=is_causal,
            is_fp8=is_fp8,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"<<<<<<{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")


def mla_decode_fwd_impl(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    logit_cap: float = 0.0,
    work_meta_data: Optional[torch.Tensor] = None,
    work_info_set: Optional[torch.Tensor] = None,
    work_indptr: Optional[torch.Tensor] = None,
    reduce_indptr: Optional[torch.Tensor] = None,
    reduce_final_map: Optional[torch.Tensor] = None,
    reduce_partial_map: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
) -> None:
    logits, lse = torch_mla_extend(
        q.clone(),
        kv_buffer.view(kv_buffer.shape[0] * kv_buffer.shape[1], 1, -1).clone(),
        qo_indptr.clone(),
        kv_indptr.clone(),
        kv_indices.clone(),
        sm_scale,
        o.size(2),
        kv_buffer.shape[-1] - o.size(2),
        dtype=q.dtype,
        is_causal=True,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    from aiter.mla import mla_decode_fwd

    mla_decode_fwd(q,
                   kv_buffer.view(-1, 1, 1, q.shape[-1]),
                   o,
                   qo_indptr,
                   kv_indptr,
                   kv_indices,
                   kv_last_page_lens,
                   max_seqlen_qo,
                   sm_scale=sm_scale,
                   logit_cap=logit_cap,
                   work_meta_data=work_meta_data,
                   work_indptr=work_indptr,
                   work_info_set=work_info_set,
                   reduce_indptr=reduce_indptr,
                   reduce_final_map=reduce_final_map,
                   reduce_partial_map=reduce_partial_map,
                   q_scale=q_scale,
                   kv_scale=kv_scale)

    cal_diff(logits, o, "out", True)


def mla_decode_fwd_fake(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    logit_cap: float = 0.0,
    work_meta_data: Optional[torch.Tensor] = None,
    work_info_set: Optional[torch.Tensor] = None,
    work_indptr: Optional[torch.Tensor] = None,
    reduce_indptr: Optional[torch.Tensor] = None,
    reduce_final_map: Optional[torch.Tensor] = None,
    reduce_partial_map: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
) -> None:
    pass


if current_platform.is_rocm():
    if is_torch_equal_or_newer("2.7.0"):
        tags = ()
    else:
        tags = (torch.Tag.needs_fixed_stride_order, ),
    direct_register_custom_op(op_name="rocm_aiter_mla_decode_fwd",
                              op_func=mla_decode_fwd_impl,
                              mutates_args=["o"],
                              fake_impl=mla_decode_fwd_fake,
                              tags=tags)
