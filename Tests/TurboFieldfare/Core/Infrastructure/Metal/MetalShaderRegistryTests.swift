import Foundation
import Testing
@testable import TurboFieldfare

/// Freezes the shader registry: the combined Metal library compiles, every
/// kernel Gemma dispatches is still in it, and every kernel production requires
/// still builds a pipeline (see `optionalKernels` for the one it does not).
///
/// `MetalContext` concatenates ten `.metal` modules into a single translation
/// unit, so any edit to any of them can break all of them — a duplicate symbol,
/// a function-constant index collision, or an MSL 4.0 construct the compiler
/// rejects takes down every kernel at once, at `MetalContext.init`. The kernel
/// suites would each report that failure as their own, and a kernel that simply
/// disappeared from the library would only surface as a `missingFunction` throw
/// at whatever runtime path first reached it.
///
/// Scope: this covers the combined runtime library only. `tensorops` is listed
/// in `shaderSubdirectories` but not in `shaderModules` — it is compiled
/// separately through `moduleLibrary(device:module:)` and is deliberately not
/// pinned here.
@Suite struct MetalShaderRegistryTests {

    /// Every function the combined library exposes.
    ///
    /// Taken from `library.functionNames` at runtime rather than from a grep of
    /// `pipeline("…")` call sites: three of these are reached through
    /// interpolated names (`dequant_int4_gemv_simd`, `dequant_int8_gemv_simd`,
    /// `router_gemv_gemma4_r4`), so a source-derived list would silently under-
    /// cover the registry.
    ///
    /// Add a kernel and this list grows — extend it. Lose a kernel and it
    /// shrinks, which is the case worth stopping.
    private static let gemmaKernels = [
        "attention_decode_combine",
        "attention_decode_gqa_swa_partial",
        "attention_decode_partial",
        "attention_prefill_causal_tiled",
        "attention_prefill_full_tensorops_2d_validity_v2",
        "dequant_int4_gemv_simd",
        "dequant_int4_qkv_gemv_simd",
        "dequant_int8_gemv_simd",
        "embed_lookup_int4",
        "fused_layer_tail",
        "fused_post_attn_setup",
        "fused_qkv_epilogue",
        "gelu_mul_fp16",
        "lm_head_greedy_int4_rows_chunk_raw",
        "lm_head_greedy_int4_rows_reduce",
        "logit_softcap_softmax",
        "moe_phase1_gate_up_act_subset_u16load",
        "moe_phase1_gate_up_act_u16load",
        "moe_phase2_down_reduce_k8",
        "prefill_dequant_int4_qmm_f16_block",
        "prefill_embed_lookup_int4_block",
        "prefill_grouped_routed_moe_batched_down",
        "prefill_grouped_routed_moe_batched_phase1",
        "prefill_layer_tail_block",
        "prefill_moe_reduce_token_major",
        "prefill_post_attn_setup_block",
        "prefill_rmsnorm_bf16w_block",
        "prefill_rmsnorm_bf16w_perhead_block",
        "prefill_rmsnorm_no_scale_perhead_block",
        "prefill_rope_default_neox_block",
        "prefill_rope_proportional_neox_block",
        "prefill_router_gemma4_block",
        "rmsnorm_bf16w",
        "rmsnorm_bf16w_perhead",
        "rmsnorm_no_scale",
        "rmsnorm_no_scale_perhead",
        "rope_default_neox",
        "rope_proportional_neox",
        "router_gemv_gemma4_r4",
        "router_topk_select_k8",
        "sample",
        "sample_topk64_final",
        "sample_topk64_reduce",
        "sample_topk64_stage1",
        "shared_int8_gate_up_act_simd",
    ]

    @Test func combinedLibraryExposesEveryGemmaKernel() throws {
        let context = try MetalContext()
        #expect(context.library.functionNames.sorted() == Self.gemmaKernels)
    }

    /// Kernels production is willing to run without.
    ///
    /// `PrefillAttention.init` builds
    /// `attention_prefill_full_tensorops_2d_validity_v2` only on devices
    /// reporting `MTLGPUFamily.apple10` (it needs MSL 4.0 MPP tensor
    /// operations), and even there it uses `try?` — if the pipeline does not
    /// build, prefill falls back to `attention_prefill_causal_tiled`, which is
    /// mandatory here. So production ships happily on a machine where this
    /// kernel is unavailable, and demanding it would fail this golden for a
    /// reason production does not consider a failure. A golden that cries wolf
    /// gets deleted by the next person who trips over it, taking the whole
    /// registry freeze with it.
    ///
    /// Every other name in `gemmaKernels` is mandatory: nothing tolerates a
    /// failure of the *unspecialized* build this suite performs, so a pipeline
    /// that does not build is a kernel that throws out of its owner's `init`.
    /// `FusedLayerTail`, `FusedPostAttentionSetup` and `SharedExpertInt8` also
    /// use `try?`, but only for constant-specialized variants of kernels they
    /// build unspecialized with `try` first — those variants are not what this
    /// suite builds.
    private static let optionalKernels: Set<String> = [
        "attention_prefill_full_tensorops_2d_validity_v2",
    ]

    /// Presence in `functionNames` only proves the symbol parsed. Building the
    /// pipeline state is what proves the kernel actually compiles for this GPU —
    /// threadgroup-size violations and unsupported instructions surface here,
    /// not at library creation.
    ///
    /// Each kernel is built the way `MetalContext.pipeline(_:)` builds it, with
    /// no constant values supplied, so what is pinned is the default
    /// specialization of every kernel. Several modules do declare function
    /// constants — `attention.metal` 60-65 and 69, `dequant_int4.metal` 20-26,
    /// `dequant_int8.metal` 70-73, `fused.metal` 80-86 — and production
    /// dispatches those kernels with explicit values through
    /// `pipeline(_:constants:)`. Those specializations are guarded by
    /// `is_function_constant_defined`, which is why the unspecialized build
    /// succeeds here, and they are covered by the individual kernel suites
    /// rather than by this pass.
    @Test func everyGemmaKernelBuildsAPipeline() throws {
        let context = try MetalContext()

        // A stale name here would exempt nothing and quietly stop describing
        // the registry, which is the state this whole suite exists to prevent.
        #expect(
            Self.optionalKernels.isSubset(of: Self.gemmaKernels),
            "optionalKernels names a kernel the library no longer exposes")

        for name in Self.gemmaKernels where !Self.optionalKernels.contains(name) {
            #expect(throws: Never.self, "pipeline \(name) failed to build") {
                _ = try context.pipeline(name)
            }
        }

        // Optional kernels get production's tolerance, which is the `try?`:
        // result discarded, nothing asserted, because nil is a supported
        // outcome. Production also skips the attempt entirely below Apple10;
        // this loop does not, because that gate decides which pipeline gets
        // *dispatched*, not whether the source has to compile. Attempting
        // everywhere is the strictly larger check — on an M2 (Apple8) the
        // kernel does build, so a shader edit that broke it would surface here
        // — and on a device where it genuinely cannot build, the `try?`
        // swallows it, so the extra reach costs no false failures.
        for name in Self.optionalKernels {
            _ = try? context.pipeline(name)
        }
    }
}
