#!/usr/bin/env python3
"""Patch deepseek_v2.py to add logging when fused shared experts activate."""
import pathlib
import sys

import vllm

vllm_pkg = pathlib.Path(vllm.__file__).parent
target = vllm_pkg / "model_executor" / "models" / "deepseek_v2.py"

code = target.read_text()

OLD = """\
        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.is_fusion_moe_shared_experts_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:"""

NEW = """\
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
        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:"""

if OLD not in code:
    print(f"WARNING: patch target not found in {target}", file=sys.stderr)
    sys.exit(0)

code = code.replace(OLD, NEW, 1)
target.write_text(code)
print(f"Patched {target}")
