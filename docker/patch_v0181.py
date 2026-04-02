#!/usr/bin/env python3
"""
Apply patches on top of vllm 0.18.1 for:
  1. Fused shared experts logging (deepseek_v2.py)
  2. Persistent MLA kernel from AITER - PR #36574 / commit 978fc18
     (_aiter_ops.py + rocm_aiter_mla.py)
  3. MLA FP8 PS + CUDA graphs accuracy fix - PR #38719 / commit 30721dc
     (rocm_aiter_mla.py + gpu_model_runner.py)
"""
import pathlib
import sys
import textwrap

import vllm

PKG = pathlib.Path(vllm.__file__).parent
ok = True


def patch(path: pathlib.Path, old: str, new: str, label: str):
    global ok
    code = path.read_text()
    if old not in code:
        print(f"WARNING: [{label}] target not found in {path}", file=sys.stderr)
        ok = False
        return
    code = code.replace(old, new, 1)
    path.write_text(code)
    print(f"  [{label}] patched {path.name}")


# ── 1. Fused shared experts logging (deepseek_v2.py) ─────────────────
target = PKG / "model_executor" / "models" / "deepseek_v2.py"
patch(
    target,
    textwrap.dedent("""\
        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
                self.is_fusion_moe_shared_experts_enabled = (
                    rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
                )
                if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:"""),
    textwrap.dedent("""\
        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
                self.is_fusion_moe_shared_experts_enabled = (
                    rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
                )
                if self.is_fusion_moe_shared_experts_enabled and config.n_shared_experts:
                    logger.info(
                        "ROCm AITER fused shared experts enabled for %s "
                        "(n_routed=%d, n_shared=%d, topk=%d)",
                        prefix,
                        config.n_routed_experts,
                        config.n_shared_experts,
                        config.num_experts_per_tok,
                    )
                if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:"""),
    "fused-se-logging",
)

# ── 2. Persistent MLA kernel – _aiter_ops.py (978fc18) ───────────────
aiter_ops = PKG / "_aiter_ops.py"

# 2a. Add work_meta_data params to _rocm_aiter_mla_decode_fwd_impl
patch(
    aiter_ops,
    """\
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> None:
    from aiter.mla import mla_decode_fwd""",
    """\
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    work_meta_data: torch.Tensor | None = None,
    work_indptr: torch.Tensor | None = None,
    work_info_set: torch.Tensor | None = None,
    reduce_indptr: torch.Tensor | None = None,
    reduce_final_map: torch.Tensor | None = None,
    reduce_partial_map: torch.Tensor | None = None,
) -> None:
    from aiter.mla import mla_decode_fwd""",
    "persistent-mla-impl-sig",
)

# 2b. Add work_meta_data kwargs forwarding before mla_decode_fwd call
patch(
    aiter_ops,
    """\
    kwargs["kv_scale"] = kv_scale

    mla_decode_fwd(""",
    """\
    kwargs["kv_scale"] = kv_scale

    if work_meta_data is not None:
        assert work_indptr is not None, (
            "work_indptr must be provided with work_meta_data"
        )
        assert work_info_set is not None, (
            "work_info_set must be provided with work_meta_data"
        )
        assert reduce_indptr is not None, (
            "reduce_indptr must be provided with work_meta_data"
        )
        assert reduce_final_map is not None, (
            "reduce_final_map must be provided with work_meta_data"
        )
        assert reduce_partial_map is not None, (
            "reduce_partial_map must be provided with work_meta_data"
        )
        kwargs["work_meta_data"] = work_meta_data
        kwargs["work_indptr"] = work_indptr
        kwargs["work_info_set"] = work_info_set
        kwargs["reduce_indptr"] = reduce_indptr
        kwargs["reduce_final_map"] = reduce_final_map
        kwargs["reduce_partial_map"] = reduce_partial_map

    mla_decode_fwd(""",
    "persistent-mla-impl-kwargs",
)

# 2c. Add params to _rocm_aiter_mla_decode_fwd_fake
patch(
    aiter_ops,
    """\
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> None:
    pass""",
    """\
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    work_meta_data: torch.Tensor | None = None,
    work_indptr: torch.Tensor | None = None,
    work_info_set: torch.Tensor | None = None,
    reduce_indptr: torch.Tensor | None = None,
    reduce_final_map: torch.Tensor | None = None,
    reduce_partial_map: torch.Tensor | None = None,
) -> None:
    pass""",
    "persistent-mla-fake-sig",
)

# 2d. Add params to the public mla_decode_fwd wrapper
patch(
    aiter_ops,
    """\
        q_scale: torch.Tensor | None = None,
        kv_scale: torch.Tensor | None = None,
    ):
        torch.ops.vllm.rocm_aiter_mla_decode_fwd(""",
    """\
        q_scale: torch.Tensor | None = None,
        kv_scale: torch.Tensor | None = None,
        work_meta_data: torch.Tensor | None = None,
        work_indptr: torch.Tensor | None = None,
        work_info_set: torch.Tensor | None = None,
        reduce_indptr: torch.Tensor | None = None,
        reduce_final_map: torch.Tensor | None = None,
        reduce_partial_map: torch.Tensor | None = None,
    ):
        torch.ops.vllm.rocm_aiter_mla_decode_fwd(""",
    "persistent-mla-public-sig",
)

# 2e. Add kwargs to the public mla_decode_fwd call site
patch(
    aiter_ops,
    """\
            q_scale=q_scale,
            kv_scale=kv_scale,
        )

    @staticmethod""",
    """\
            q_scale=q_scale,
            kv_scale=kv_scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

    @staticmethod""",
    "persistent-mla-public-call",
)

# ── 3. Persistent MLA + FP8 fix – rocm_aiter_mla.py ─────────────────
mla = PKG / "v1" / "attention" / "backends" / "mla" / "rocm_aiter_mla.py"

# 3a. Add CommonAttentionMetadata import
patch(
    mla,
    "from vllm.v1.attention.backend import AttentionCGSupport, AttentionLayer, MultipleOf",
    """\
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
    MultipleOf,
)""",
    "mla-import-commonattnmeta",
)

# 3b. Add @dataclass and persistent metadata fields to AiterMLAMetadata
patch(
    mla,
    """\
class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass""",
    """\
@dataclass
class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None""",
    "mla-metadata-fields",
)

# 3c. Add persistent MLA metadata buffers in __init__ + FP8 dtype detection
patch(
    mla,
    """\
            max_num_pages, dtype=torch.int32, device=device
        )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():""",
    """\
            max_num_pages, dtype=torch.int32, device=device
        )

        from aiter import dtypes, get_mla_metadata_info_v1

        _cache_dtype_str = getattr(
            vllm_config.cache_config, "cache_dtype", "auto"
        )
        if _cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            self._metadata_kv_dtype = dtypes.fp8
        else:
            self._metadata_kv_dtype = dtypes.bf16
        self._metadata_q_dtype = self._metadata_kv_dtype

        self._num_attention_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        q_dtype = self.decode_attn_out_dtype
        kv_cache_dtype_str = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            kv_cache_dtype_str = "fp8"
        else:
            kv_cache_dtype_str = "bf16"
        kv_dtype = dtypes.d_dtypes.get(kv_cache_dtype_str, dtypes.bf16)
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max_num_reqs,
            1,
            self._num_attention_heads,
            self._metadata_q_dtype,
            self._metadata_kv_dtype,
            is_sparse=False,
            fast_mode=True,
        )
        self._mla_work_meta_data = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        self._mla_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        self._mla_work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        )
        self._mla_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        self._mla_reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        self._mla_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_type,
            device=device,
        )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():""",
    "mla-persistent-buffers",
)

# 3d. Add get_mla_metadata_v1 call in _build_decode
patch(
    mla,
    """\
        else:
            qo_indptr = torch.arange(
                0, num_reqs + 1, step=1, dtype=torch.int32, device=device
            )

        attn_metadata = AiterMLADecodeMetadata(""",
    """\
        else:
            qo_indptr = torch.arange(
                0, num_reqs + 1, step=1, dtype=torch.int32, device=device
            )

        from aiter import get_mla_metadata_v1

        get_mla_metadata_v1(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_last_page_len,
            self._num_attention_heads,
            1,
            True,
            self._mla_work_meta_data,
            self._mla_work_info_set,
            self._mla_work_indptr,
            self._mla_reduce_indptr,
            self._mla_reduce_final_map,
            self._mla_reduce_partial_map,
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=max_qo_len,
            uni_seqlen_qo=max_qo_len,
            fast_mode=True,
            dtype_q=self._metadata_q_dtype,
            dtype_kv=self._metadata_kv_dtype,
            max_split_per_batch=16,
        )

        attn_metadata = AiterMLADecodeMetadata(""",
    "mla-build-decode-metadata",
)

# 3e. Add build() override to pass persistent metadata
patch(
    mla,
    """\
        return attn_metadata


@triton.jit
def _copy_page_indices_kernel(""",
    """\
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AiterMLAMetadata:
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        attn_metadata.work_meta_data = self._mla_work_meta_data
        attn_metadata.work_indptr = self._mla_work_indptr
        attn_metadata.work_info_set = self._mla_work_info_set
        attn_metadata.reduce_indptr = self._mla_reduce_indptr
        attn_metadata.reduce_final_map = self._mla_reduce_final_map
        attn_metadata.reduce_partial_map = self._mla_reduce_partial_map
        return attn_metadata


@triton.jit
def _copy_page_indices_kernel(""",
    "mla-build-override",
)

# 3f. Pass persistent metadata in forward_mqa decode call
patch(
    mla,
    """\
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
        )

        if self._needs_head_repeat:""",
    """\
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
            work_meta_data=attn_metadata.work_meta_data,
            work_indptr=attn_metadata.work_indptr,
            work_info_set=attn_metadata.work_info_set,
            reduce_indptr=attn_metadata.reduce_indptr,
            reduce_final_map=attn_metadata.reduce_final_map,
            reduce_partial_map=attn_metadata.reduce_partial_map,
        )

        if self._needs_head_repeat:""",
    "mla-forward-persistent-kwargs",
)

# ── 4. FP8 dummy-run slot_mapping fix – gpu_model_runner.py (PR #38719) ──
gmr = PKG / "v1" / "worker" / "gpu_model_runner.py"
patch(
    gmr,
    """\
            ubatch_slices=ubatch_slices_padded,
        )

        # _dummy_run shares pinned CPU buffers""",
    """\
            ubatch_slices=ubatch_slices_padded,
        )

        # During dummy runs, slot_mapping contains uninitialized values
        # (zeros from buffer creation) because no real requests have been
        # scheduled. Writing FP8 KV data to those slot positions corrupts
        # the KV cache. Fill the entire slot_mapping with PAD_SLOT_ID (-1)
        # so that concat_and_cache_mla skips the writes.
        if slot_mappings_by_group is not None:
            for sm in slot_mappings_by_group.values():
                sm.fill_(-1)

        # _dummy_run shares pinned CPU buffers""",
    "fp8-dummy-run-slot-mapping",
)

if ok:
    print("All patches applied successfully.")
else:
    print("SOME PATCHES FAILED – check warnings above.", file=sys.stderr)
    sys.exit(1)
