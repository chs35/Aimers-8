# vLLM 0.14.1 — KV Cache Quantization Auto-Detection Analysis

## Executive Summary

**In vLLM 0.14.1, KV cache quantization IS automatically detected from ModelOpt's `hf_quant_config.json` — the `--kv-cache-dtype` flag is NOT required for ModelOpt-quantized models.** This is the result of three sequential PRs (#29980 → revert → #30785 → #31895) that were all merged before the v0.14.0 release. However, this auto-detection is limited to ModelOpt quantization configs only; compressed-tensors models still require explicit `--kv-cache-dtype fp8`.

---

## PR Dependency Chain & Timeline

```
PR #29980 (Dec 3–12, 2025) — Initial implementation
    │  "Load kv-cache dtype from hf_quant_config.json automatically"
    │   Author: danielafrimi (NVIDIA)
    │   Reviewers: ywang96, mgoin (approved)
    │   Status: MERGED → then REVERTED
    │
    ├── PR #30653 (Dec 14, 2025) — Revert
    │    Reason: Broke CUTLASS_MLA CI tests
    │    "CUTLASS_MLA does not support Fp8 KV cache"
    │    (dtype assertion failed: q_nope.dtype == kv_c_and_k_pe_cache.dtype)
    │
    ├── PR #30785 (Dec 16–17, 2025) — Fixed re-implementation
    │    "Load kv-cache dtype from hf_quant_config.json automatically (fix for reverted PR)"
    │    Author: danielafrimi (NVIDIA)
    │    Reviewer: robertgshaw2-redhat (approved)
    │    Key fix: Early resolution in arg_utils.py so CacheConfig propagates correctly
    │    Status: MERGED into main
    │
    └── PR #31895 (Jan 7–10, 2026) — Schema evolution support
         "Update modelopt KV cache quantization resolution to new scheme"
         Author: roikoren755 (NVIDIA)
         Reviewer: ywang96 (approved)
         Handles new ModelOpt dict-based kv_cache_scheme format
         Status: MERGED into main → included in v0.14.0 release
```

All three PRs landed BEFORE v0.14.0 (Jan 17, 2026). PR #31895 is explicitly listed in the v0.14.0 release notes under Quantization: "ModelOpt KV cache quantization update (#31895)".

---

## Detailed PR Analysis

### PR #29980: The Original Auto-Detection

**Problem:** ModelOpt's `hf_quant_config.json` specifies `"kv_cache_quant_algo": "FP8"`, but vLLM ignored this and loaded the unquantized KV cache dtype. Users had to explicitly pass `--kv-cache-dtype fp8` even for models pre-calibrated with FP8 KV cache.

**Solution:** Modified `kv_cache_dtype_str_to_dtype()` in `vllm/utils/torch_utils.py` to check the model's quantization config when `kv_cache_dtype == "auto"`. If `kv_cache_quant_algo` is found (e.g., `"FP8"`), it automatically resolves to the correct FP8 dtype.

**Key reviewer comment from @mgoin (vLLM maintainer):**
> "This change is good with me, thank you. I've wanted to change the default of 'auto' to be unquantized unless the model checkpoint has calibrated scales, which this achieves for modelopt. So in line with what I want!"

**What went wrong:** The implementation resolved `kv_cache_dtype` too late in the pipeline. When loading NVIDIA's DeepSeek NVFP4 model, the CUTLASS MLA attention backend received FP8 KV cache tensors but expected BF16, causing an assertion failure:
```
assert q_nope.dtype == q_pe.dtype == kv_c_and_k_pe_cache.dtype
```

### PR #30785: The Correct Fix

**Root cause of the revert:** The original PR resolved the dtype inside `kv_cache_dtype_str_to_dtype()` which runs too late. The attention backend selection logic had already been finalized without knowing the actual KV cache dtype.

**Fix:** Moved the ModelOpt config parsing **earlier** — into `arg_utils.py` when creating `CacheConfig`. This way:
1. The resolved `kv_cache_dtype` value is set in `CacheConfig` **before** any downstream systems read it
2. Attention backend selection sees the correct dtype
3. Backends that don't support FP8 KV cache (like CUTLASS MLA) can properly reject/adapt

**Code location:** `vllm/engine/arg_utils.py` — at `CacheConfig` construction time, reads `hf_quant_config.json` → finds `kv_cache_quant_algo` → maps `"FP8"` to `"fp8"` → sets `cache_dtype` accordingly.

### PR #31895: New ModelOpt Schema Support

**Problem:** Newer versions of ModelOpt changed the quantization config format. Previously:
```json
{
  "kv_cache_quant_algo": "FP8"
}
```

New format stores under `kv_cache_scheme` as either a string (like before) or a dictionary:
```json
{
  "kv_cache_scheme": {
    "dynamic": false,
    "num_bits": 8,
    "type": "float"
  }
}
```

**Solution:** Updated the resolution logic in `vllm/utils/torch_utils.py` to handle both formats:
- If `kv_cache_scheme` is a string → treat as before (map `"FP8"` to `"fp8"`)
- If `kv_cache_scheme` is a dict → validate: `dynamic is False`, `num_bits == 8`, `type == "float"` → resolve to `"fp8"`

**Code (from PR review):**
```python
if (
    kv_algo.get("dynamic") is False
    and kv_algo.get("num_bits") == 8
    and kv_algo.get("type") == "float"
):
    # Resolve to fp8
```

---

## Config Schema Analysis: When Auto-Detection Works

### ✅ Auto-detected (no `--kv-cache-dtype` needed)

| Quantization Method | Config File | Config Key | Supported Since |
|---|---|---|---|
| ModelOpt (legacy) | `hf_quant_config.json` | `"kv_cache_quant_algo": "FP8"` | v0.14.0 (PR #30785) |
| ModelOpt (new) | `hf_quant_config.json` | `"kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"}` | v0.14.0 (PR #31895) |
| DeepSeekV3.2 | Architecture default | Hardcoded in platform config | v0.13.x+ |

### ❌ NOT auto-detected (still requires `--kv-cache-dtype fp8`)

| Quantization Method | Config File | Config Key | Status |
|---|---|---|---|
| compressed-tensors (llm-compressor) | `config.json` → `quantization_config` | `kv_cache_scheme` inside CT format | NOT auto-detected |
| Manual FP8 KV cache | None | User wants runtime quantization | Requires explicit flag |
| `calculate_kv_scales=True` | None | Dynamic scale estimation | Requires explicit flag |

---

## `CacheConfig.cache_dtype` Value Resolution (v0.14.1)

```
User passes --kv-cache-dtype (or leaves "auto")
          │
          ▼
    ┌─────────────┐
    │ "auto"?      │──── No ──→ Use explicit value (fp8/fp8_e4m3/fp8_e5m2/etc.)
    └─────────────┘
          │ Yes
          ▼
    ┌─────────────────────────┐
    │ Check hf_quant_config   │  (arg_utils.py, early in CacheConfig creation)
    │ for ModelOpt config     │
    └─────────────────────────┘
          │
          ├── Found "kv_cache_quant_algo": "FP8" ──→ Resolve to "fp8"
          ├── Found "kv_cache_scheme": dict{...}  ──→ Validate & resolve to "fp8"
          ├── Found "kv_cache_scheme": "FP8"      ──→ Resolve to "fp8"
          │
          └── Not found ──→ Check architecture defaults
                │
                ├── DeepSeekV3.2 ──→ Override to "fp8"
                │
                └── All others ──→ Resolve to model_config.dtype (bf16/fp16)
                                   (No quantization applied)
```

---

## v0.14.0 / v0.14.1 Changelog: KV Cache Quantization Entries

### v0.14.0 (Jan 17, 2026)
Under **Quantization** section:
- "**ModelOpt KV cache quantization update (#31895)**" — the schema evolution PR
- "ModelOpt FP8 variants (FP8_PER_CHANNEL_PER_TOKEN, FP8_PB_WO) (#30957)"
- "compressed-tensors 0.13.0 (#30799)"

Note: PR #30785 (the core auto-detection fix) is NOT explicitly listed in the release notes despite being merged Dec 17. It may be bundled under the #31895 entry since #31895 built directly on top of #30785's changes.

### v0.14.1 (Jan 24, 2026)
- Described as "a patch release on top of v0.14.0 to address a few security and memory leak fixes"
- **No KV cache quantization changes** in v0.14.1
- Contains only: DeepGEMM scale shapes fix, DP+MoE inference fix, UCX memory leak fix, structured output fix

---

## Known Issues in v0.14.x

### Bug #32732: Regression on Blackwell (RTX Pro 6000)
PR #30785's auto-detection causes `nvidia/DeepSeek-R1-0528-NVFP4` to fail with "No valid attention backend found" on Blackwell GPUs. The auto-detected FP8 KV cache dtype conflicts with attention backend requirements. This is the same class of issue that caused the original PR #29980 revert — certain attention backends (CUTLASS MLA, some Blackwell-specific backends) cannot handle FP8 KV cache.

### Issue #32843: `kv_cache_dtype="auto"` UX confusion
Feature request to make the `"auto"` resolution transparent to users by logging/displaying the actual resolved dtype. Currently, when auto-detection activates, users may not realize their KV cache is running in FP8 mode.

---

## Practical Recommendations for Our Deployment

### For ModelOpt-quantized models (e.g., NVIDIA TensorRT-LLM exports):
```bash
# No --kv-cache-dtype needed! Auto-detected from hf_quant_config.json
vllm serve <model_path>
```

### For llm-compressor / compressed-tensors models:
```bash
# MUST explicitly specify
vllm serve <model_path> --kv-cache-dtype fp8
```

### To override auto-detection (e.g., if hitting backend compatibility issues):
```bash
# Force BF16 KV cache even if model has FP8 config
vllm serve <model_path> --kv-cache-dtype bfloat16
```

### To verify what dtype is actually being used:
```bash
# Enable model inspection logging
VLLM_LOG_MODEL_INSPECTION=1 vllm serve <model_path>
```

---

## Conclusion

The three PRs together form a complete auto-detection pipeline for ModelOpt KV cache quantization in vLLM 0.14.1:

1. **PR #29980** established the principle (auto-detect from `hf_quant_config.json`)
2. **PR #30785** fixed the implementation timing (early resolution in `arg_utils.py`)
3. **PR #31895** extended compatibility (new dict-based `kv_cache_scheme` format)

**For our company's deployment on vLLM 0.14.1:**
- If using **ModelOpt-quantized models**: KV cache quantization works automatically without `--kv-cache-dtype`
- If using **compressed-tensors / llm-compressor models**: Still need explicit `--kv-cache-dtype fp8`
- Watch for **Bug #32732** if deploying on Blackwell GPUs — auto-detection may conflict with CUTLASS MLA backend
- Consider setting `--kv-cache-dtype` explicitly in production configs for clarity and to avoid surprises from auto-detection
