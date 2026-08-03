import Foundation
import Testing
@testable import TurboFieldfareRepackCore

/// Freezes the resident tensor list the planner emits for Gemma: which tensors
/// are in it, and in what order.
///
/// The order is the input to every `fileOffset` in `model_weights.bin`, so this
/// suite and `RangeCopyPlannerTests.gemmaByteLayoutIsFrozen` fail together. The
/// split is deliberate: the fingerprint proves *something* moved, this suite
/// names *which tensor* moved, which is the difference between a five-minute
/// triage and an afternoon of bisecting.
@Suite
struct RepackPlannerTests {

    /// Full-attention layers share K and V (`attention_k_eq_v`), so they ship no
    /// `v_proj` at all — the runtime reads V out of the K projection.
    ///
    /// This is worth its own assertion because the failure is quiet in both
    /// directions. A naming refactor that derives per-layer tensor names from a
    /// template rather than from what the checkpoint actually contains would
    /// reintroduce `v_proj` on layer 1; the planner would then look for a tensor
    /// the source does not have, or — worse, if it tolerated the absence —
    /// shift every subsequent offset by one slot. `SyntheticSnapshot` omits
    /// `v_proj` on its full-attention layer precisely so this is testable
    /// offline.
    @Test func fullAttentionLayerHasNoResidentVProjection() throws {
        try GemmaFrozenPlan.withPlans { repackPlan, _ in
            let names = repackPlan.resident.entries.map(\.name)

            // Layer 0 is sliding_attention: K and V are separate.
            #expect(names.contains("language_model.model.layers.0.self_attn.v_proj.weight"))

            // Layer 1 is full_attention: V is K.
            #expect(
                !names.contains("language_model.model.layers.1.self_attn.v_proj.weight"),
                """
                A v_proj entry appeared on the full-attention layer. Gemma \
                shares K and V there (attention_k_eq_v), so this tensor does \
                not exist in the checkpoint — planning it shifts every later \
                resident offset and rewrites model_weights.bin.
                """)
        }
    }

    /// The exact resident order, tensor by tensor.
    ///
    /// Produced by `RepackPlanner.lmResidentOrdering` — embedding, then each
    /// layer's slots in `slotRank` order, then the final norm. Changing
    /// `slotRank` reorders this list, which relocates every tensor in
    /// `model_weights.bin` and invalidates in-flight installs. As with the
    /// fingerprint golden: update this list only when the layout change is the
    /// deliberate point of the change, never to quiet a failure.
    @Test func residentOrderingIsFrozen() throws {
        try GemmaFrozenPlan.withPlans { repackPlan, _ in
            #expect(repackPlan.resident.entries.map(\.name) == [
                "language_model.model.embed_tokens.weight",

                // Layer 0 — sliding_attention, so v_proj is present.
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "language_model.model.layers.0.self_attn.k_proj.weight",
                "language_model.model.layers.0.self_attn.v_proj.weight",
                "language_model.model.layers.0.self_attn.o_proj.weight",
                "language_model.model.layers.0.self_attn.q_norm.weight",
                "language_model.model.layers.0.self_attn.k_norm.weight",
                "language_model.model.layers.0.router.proj.weight",
                "language_model.model.layers.0.router.scale",
                "language_model.model.layers.0.router.per_expert_scale",
                "language_model.model.layers.0.mlp.gate_proj.weight",
                "language_model.model.layers.0.mlp.up_proj.weight",
                "language_model.model.layers.0.mlp.down_proj.weight",
                "language_model.model.layers.0.input_layernorm.weight",
                "language_model.model.layers.0.post_attention_layernorm.weight",
                "language_model.model.layers.0.pre_feedforward_layernorm.weight",
                "language_model.model.layers.0.pre_feedforward_layernorm_2.weight",
                "language_model.model.layers.0.post_feedforward_layernorm.weight",
                "language_model.model.layers.0.post_feedforward_layernorm_1.weight",
                "language_model.model.layers.0.post_feedforward_layernorm_2.weight",
                "language_model.model.layers.0.layer_scalar",

                // Layer 1 — full_attention, so no v_proj.
                "language_model.model.layers.1.self_attn.q_proj.weight",
                "language_model.model.layers.1.self_attn.k_proj.weight",
                "language_model.model.layers.1.self_attn.o_proj.weight",
                "language_model.model.layers.1.self_attn.q_norm.weight",
                "language_model.model.layers.1.self_attn.k_norm.weight",
                "language_model.model.layers.1.router.proj.weight",
                "language_model.model.layers.1.router.scale",
                "language_model.model.layers.1.router.per_expert_scale",
                "language_model.model.layers.1.mlp.gate_proj.weight",
                "language_model.model.layers.1.mlp.up_proj.weight",
                "language_model.model.layers.1.mlp.down_proj.weight",
                "language_model.model.layers.1.input_layernorm.weight",
                "language_model.model.layers.1.post_attention_layernorm.weight",
                "language_model.model.layers.1.pre_feedforward_layernorm.weight",
                "language_model.model.layers.1.pre_feedforward_layernorm_2.weight",
                "language_model.model.layers.1.post_feedforward_layernorm.weight",
                "language_model.model.layers.1.post_feedforward_layernorm_1.weight",
                "language_model.model.layers.1.post_feedforward_layernorm_2.weight",
                "language_model.model.layers.1.layer_scalar",

                "language_model.model.norm.weight",
            ])
        }
    }

    /// Resident entries are laid out back to back after the index page, and each
    /// quantized entry keeps its scales and biases immediately behind its
    /// weights. This is what makes the order above equal the byte layout: if
    /// packing ever grew a gap or reordered the companions, the name list could
    /// stay identical while every offset moved.
    @Test func residentEntriesArePackedInOrderAfterTheIndex() throws {
        try GemmaFrozenPlan.withPlans { repackPlan, _ in
            var cursor = repackPlan.resident.indexSize
            for entry in repackPlan.resident.entries {
                #expect(entry.fileOffset == cursor, "gap or overlap before \(entry.name)")
                cursor = entry.fileOffset + entry.sizeBytes
                if entry.quantSpec != nil, entry.scaleSize > 0 {
                    #expect(entry.scaleOffset == cursor, "scales moved for \(entry.name)")
                    cursor = entry.scaleOffset + entry.scaleSize
                    #expect(entry.biasOffset == cursor, "biases moved for \(entry.name)")
                    cursor = entry.biasOffset + entry.biasSize
                }
            }
            #expect(cursor == repackPlan.resident.totalSize)
        }
    }
}
