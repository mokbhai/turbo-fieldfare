import Testing
import Foundation
@testable import TurboFieldfare

/// Value pins for `ArchConfig.lingMini2_0_4bit`.
///
/// `ModelTypesTests`' Mirror check catches a *missing* field. It cannot catch a
/// field holding the wrong number, and for Ling one wrong number is a silent
/// ten-fold GEMV mis-sizing rather than a crash — so the values are pinned here
/// against the published `inclusionAI/Ling-mini-2.0` config, transcribed rather
/// than derived from the code under test.
@Suite struct ArchConfigLingTests {

    /// THE TRAP, pinned on its own because it is the one field where both
    /// candidate values parse, both are plausible, and nothing downstream
    /// notices the wrong one.
    ///
    /// `intermediateSize` is documented and consumed as the SHARED-EXPERT FFN
    /// width, and Ling's `moe_shared_expert_intermediate_size` is 512. Ling's
    /// config also carries a key literally named `intermediate_size` = 5120,
    /// which is the DENSE layer-0 FFN width and belongs in the payload. Setting
    /// `intermediateSize = 5120` silently redefines a field the runtime already
    /// reads as the shared-expert GEMV width.
    @Test func sharedExpertWidthIsNotTheDenseWidth() {
        let a = ArchConfig.lingMini2_0_4bit
        #expect(a.intermediateSize == 512)
        guard case .bailingMoeV2(let extras) = a.variant else {
            Issue.record("lingMini2_0_4bit is not a bailingMoeV2 variant")
            return
        }
        #expect(extras.denseIntermediateSize == 5120)
        #expect(a.intermediateSize != extras.denseIntermediateSize)
    }

    @Test func lingBaselineMatchesThePublishedConfig() {
        let a = ArchConfig.lingMini2_0_4bit

        #expect(a.hiddenSize == 2048)
        #expect(a.intermediateSize == 512)
        #expect(a.moeIntermediateSize == 512)
        #expect(a.numHeads == 16)
        #expect(a.numKVHeads == 4)
        #expect(a.headDim == 128)
        #expect(a.vocabSize == 157184)
        #expect(a.fullRopeTheta == 600_000.0)
        #expect(a.partialRotaryFactor == 0.5)
        #expect(a.numLayers == 20)
        #expect(a.numExperts == 256)
        #expect(a.topKExperts == 8)
        #expect(a.tieWordEmbeddings == false)
        #expect(a.hiddenActivation == "silu")

        // No sliding window and no `layer_types`: every layer is full attention.
        // Spelled out rather than recomputed so the pin cannot agree with a
        // broken generator, and because an EMPTY mask (Gemma's `?? []` default)
        // would be indexed by layer at runtime and trap.
        #expect(a.fullAttentionLayerMask == [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ] as [UInt8])

        // Ling draws no sliding/global distinction, so the "full" head geometry
        // is the head geometry and the sliding-only fields say "none". Every one
        // of these is a definite value that `validateArch` compares for Ling
        // too — `numFullKVHeads`/`fullHeadDim` most of all, since a 100%
        // full-attention model reads exactly those.
        #expect(a.numFullKVHeads == a.numKVHeads)
        #expect(a.fullHeadDim == a.headDim)
        #expect(a.slidingWindow == 0)
        #expect(a.finalLogitSoftcap == 0.0)
        #expect(a.ropeTheta == 600_000.0)
        #expect(a.attentionKEqV == false)

        guard case .bailingMoeV2(let extras) = a.variant else {
            Issue.record("lingMini2_0_4bit is not a bailingMoeV2 variant")
            return
        }
        #expect(extras.denseIntermediateSize == 5120)
        #expect(extras.firstKDenseReplace == 1)
        #expect(extras.numSharedExperts == 1)
        #expect(extras.nGroup == 8)
        #expect(extras.topkGroup == 4)
        #expect(extras.routedScalingFactor == 2.5)
        #expect(extras.normTopkProb == true)
        #expect(extras.scoreFunction == "sigmoid")
        #expect(extras.routerEnableExpertBias == true)
        #expect(extras.useQKNorm == true)
    }

    /// **The milestone's safety property, as the consequence rather than as a
    /// list membership.**
    ///
    /// "A real Ling install must not load in a build with no BailingMoeV2
    /// kernels." Not "Ling must not appear in `supported`" — that is the
    /// mechanism, and a mechanism can be replaced by one that reads the same
    /// and enforces nothing. What has to hold is that the load is refused.
    ///
    /// The manifest below has nothing wrong with it: the published baseline,
    /// the quant block Ling really carries, every required file listed. The
    /// only reason to refuse it is that this build cannot execute it, so that
    /// is the only thing this test can be passing for.
    ///
    /// Written over `real` minus `supported` rather than over Ling by name, so
    /// the next architecture whose baseline lands before its kernels is covered
    /// the moment it is added.
    @Test func aRealLingInstallCannotLoadWithoutItsKernels() throws {
        let unexecutable = ArchConfig.real.filter { !ArchConfig.supported.contains($0) }
        #expect(!unexecutable.isEmpty,
                """
                every real architecture now has kernels. If Ling's landed, this test \
                needs a genuinely unexecutable case to guard — deleting it removes the \
                only check that kernels precede loadability.
                """)

        for architecture in unexecutable {
            let family = architecture.variant.family.rawValue
            let (dir, _) = try ManifestReaderTests.writeToyManifest(
                ["quant": ManifestReaderTests.quantBlock(architecture.variant.family)],
                config: architecture)
            defer { try? FileManager.default.removeItem(at: dir) }

            // 1. The production path. `Model.load` defaults `expecting:` to the
            //    Gemma baseline and every caller in this package takes that
            //    default, so this is what a user's install actually meets. The
            //    refusal names `family`, which is the reason-reporting path
            //    doing its job rather than a bare no.
            #expect {
                _ = try ManifestReader.load(directoryURL: dir,
                                            expecting: .gemma4_26B_A4B)
            } throws: { error in
                guard case let ModelError.archMismatch(field, _, actual) = error else {
                    return false
                }
                return field == "family" && actual == family
            }

            // 2. The path a caller could talk itself into: offering the
            //    architecture its OWN baseline. Every field-by-field comparison
            //    passes and the quant block is correct, so nothing else in the
            //    loader has an objection left — this is exactly where "no
            //    kernels" has to be the thing that refuses, and it is the case
            //    a test written only against the Gemma baseline would miss.
            #expect {
                _ = try ManifestReader.load(directoryURL: dir,
                                            expecting: architecture)
            } throws: { error in
                guard case let ModelError.architectureNotExecutable(reported) = error
                else { return false }
                return reported == family
            }
        }
    }

    /// The mechanism behind the property above. If this fails, Ling was added
    /// to `supported` — that is a finding, not a test to update.
    ///
    /// `supported` means "this build ships kernels for these bytes". No
    /// BailingMoeV2 kernels exist yet, so an entry here would let a Ling install
    /// complete, pass every gate, and then trap at the first attention dispatch
    /// with nothing pointing back at the cause. It is added alongside the
    /// kernels.
    @Test func lingIsDeliberatelyNotInSupported() {
        #expect(!ArchConfig.supported.contains(ArchConfig.lingMini2_0_4bit))
        #expect(ArchConfig.supported == [ArchConfig.gemma4_26B_A4B])

        // The consequence, asserted so it is a decision and not an accident: a
        // Ling manifest cannot be matched into an executable architecture.
        let lingArch = ManifestArch(from: ArchConfig.lingMini2_0_4bit)
        #expect(ManifestReader.matchArch(lingArch, against: ArchConfig.supported) == nil)
    }

    /// **The gate keys on the family, so the family list is what has to be
    /// right.**
    ///
    /// `executableFamilies` is derived from `supported` rather than listed, so
    /// this cannot drift from it — but it is pinned anyway because it is the
    /// thing `ManifestReader.validate` consults, and because "gemma4 only" is
    /// the standing statement that this binary contains no BailingMoeV2
    /// kernels. If a family appears here, kernels for it exist.
    @Test func onlyGemmaIsExecutable() {
        #expect(ArchConfig.executableFamilies == [.gemma4])
        #expect(!ArchConfig.executableFamilies.contains(.bailingMoeV2))
        // Derived, not hand-written: every supported entry's family is in it,
        // and nothing else is.
        #expect(ArchConfig.executableFamilies
                == Set(ArchConfig.supported.map { $0.variant.family }))
    }

    /// Being out of `supported` must not also excuse Ling from the checks that
    /// apply to a real install. `real` is what the mandatory-`quant` gate keys
    /// off, and `supported` being a subset of it is what makes that gate no
    /// looser for Gemma than the one it replaced.
    @Test func lingIsInTheRealArchitectureList() {
        #expect(ArchConfig.real.contains(ArchConfig.lingMini2_0_4bit))
        #expect(ArchConfig.real.contains(ArchConfig.gemma4_26B_A4B))
        for architecture in ArchConfig.supported {
            #expect(ArchConfig.real.contains(architecture))
        }

        let lingArch = ManifestArch(from: ArchConfig.lingMini2_0_4bit)
        #expect(ManifestReader.matchArch(lingArch, against: ArchConfig.real) != nil)
    }

    /// Ling's router is 4-bit and Gemma's is 8-bit, so the accepted widths are
    /// per-architecture. Pinned here as well as in the rejection tests because
    /// this is the table itself rather than one path through it.
    @Test func acceptedRouterWidthsDifferByFamily() {
        #expect(ArchConfig.lingMini2_0_4bit.variant.acceptedQuantBits.router == [4])
        #expect(ArchConfig.gemma4_26B_A4B.variant.acceptedQuantBits.router == [8])
    }
}
