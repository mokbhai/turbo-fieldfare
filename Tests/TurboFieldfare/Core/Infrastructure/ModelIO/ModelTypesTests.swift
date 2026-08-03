import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct ModelTypesTests {

    /// Pins every field of the Gemma baseline, not just the headline ones.
    ///
    /// `ArchConfig` is compared field-by-field against `manifest.json` at load
    /// time and is the dispatch input for attention, RoPE and MoE, so a wrong
    /// value here does not fail loudly — it either rejects a correct install or
    /// runs the wrong kernel shape. The uncovered fields are the dangerous ones:
    /// a refactor that reorganises architectures into families and variants
    /// touches all of them, and the head/RoPE values in particular have no other
    /// assertion anywhere in the suite.
    ///
    /// The field-name inventory below is part of the pin. Values alone go stale
    /// silently: add a 22nd field with a wrong default and every #expect here
    /// still passes. Asserting the label list forces whoever adds a field to
    /// come here and pin it.
    @Test func archConfigGemma4BaselineMatchesDocs() {
        let a = ArchConfig.gemma4_26B_A4B

        #expect(Mirror(reflecting: a).children.compactMap(\.label) == [
            "hiddenSize",
            "intermediateSize",
            "moeIntermediateSize",
            "numHeads",
            "numKVHeads",
            "numFullKVHeads",
            "headDim",
            "fullHeadDim",
            "vocabSize",
            "slidingWindow",
            "finalLogitSoftcap",
            "ropeTheta",
            "fullRopeTheta",
            "partialRotaryFactor",
            "numLayers",
            "numExperts",
            "topKExperts",
            "tieWordEmbeddings",
            "attentionKEqV",
            "fullAttentionLayerMask",
            "hiddenActivation",
        ], "ArchConfig gained or lost a field; pin its Gemma value below")

        #expect(a.hiddenSize == 2816)
        #expect(a.intermediateSize == 2112)
        #expect(a.moeIntermediateSize == 704)
        #expect(a.numHeads == 16)
        #expect(a.numKVHeads == 8)
        #expect(a.numFullKVHeads == 2)
        #expect(a.headDim == 256)
        #expect(a.fullHeadDim == 512)
        #expect(a.vocabSize == 262144)
        #expect(a.slidingWindow == 1024)
        #expect(a.finalLogitSoftcap == 30.0)
        #expect(a.ropeTheta == 10_000.0)
        #expect(a.fullRopeTheta == 1_000_000.0)
        #expect(a.partialRotaryFactor == 0.25)
        #expect(a.numLayers == 30)
        #expect(a.numExperts == 128)
        #expect(a.topKExperts == 8)
        #expect(a.tieWordEmbeddings == true)
        #expect(a.attentionKEqV == true)
        #expect(a.hiddenActivation == "gelu_pytorch_tanh")

        // Every sixth layer from 5 is full-attention; the rest are sliding. The
        // literal is spelled out rather than recomputed so the test cannot agree
        // with a broken generator.
        #expect(a.fullAttentionLayerMask == [
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
        ] as [UInt8])

        // Gemma is the only architecture with kernels in this build. An entry
        // added here without them installs cleanly and then traps at the first
        // attention dispatch.
        #expect(ArchConfig.supported == [ArchConfig.gemma4_26B_A4B])
    }

    @Test func modelErrorDescriptionsContainKeyFacts() {
        let e1 = ModelError.archMismatch(field: "hiddenSize", expected: "2816", actual: "4096")
        #expect(e1.description.contains("2816") && e1.description.contains("4096"))
        let e2 = ModelError.unsupportedVersion(major: 2, minor: 0)
        #expect(e2.description.contains("2"))
        let e3 = ModelError.checksumMismatch(file: "model_weights.bin")
        #expect(e3.description.contains("model_weights.bin"))
    }
}
