import Foundation
import Testing

@testable import TurboFieldfareRepackCore

/// The writer half of the family contract.
///
/// The reader defaults a missing `arch.family` to `gemma4`; this suite pins the
/// other end — that the writer keeps omitting it for Gemma, and emits it plus
/// the whole payload for BailingMoeV2.
@Suite struct GTurboJSONArchFamilyTests {

    /// **Gemma's manifest bytes must not move.**
    ///
    /// `manifest.json`'s SHA-256 is bound by `VerifiedInstallReceipt`, and an
    /// install already on a user's disk cannot be rewritten. Emitting
    /// `"family": "gemma4"` unconditionally would be self-consistent for new
    /// installs and would still make every freshly repacked Gemma manifest
    /// differ from every shipped one. Omitting it also keeps
    /// `AppModelInstallFixture`'s hand-written, family-less arch dict a live
    /// proof of the reader's default rather than a stale relic.
    @Test func gemmaManifestCarriesNoFamilyKey() throws {
        let archDict = try Self.encodedArchDict(overridingArchWith: nil)
        #expect(archDict["family"] == nil)
        // Exactly the 21 keys shipped manifests have, and no others: a payload
        // key leaking onto the Gemma branch changes its bytes just as surely as
        // `family` would. Spelled out because these strings are the wire format
        // the reader's `CodingKeys` must match character for character.
        #expect(archDict.keys.sorted() == [
            "attentionKEqV",
            "ffnIntermediate",
            "finalLogitSoftcap",
            "fullAttentionLayerMask",
            "fullHeadDim",
            "fullRopeTheta",
            "headDim",
            "hiddenActivation",
            "hiddenSize",
            "moeIntermediateSize",
            "numExperts",
            "numFullKVHeads",
            "numHeads",
            "numKVHeads",
            "numLayers",
            "partialRotaryFactor",
            "ropeTheta",
            "slidingWindow",
            "tieWordEmbeddings",
            "topKExperts",
            "vocabSize",
        ])
    }

    @Test func bailingManifestCarriesFamilyAndPayload() throws {
        let ling = try Self.lingArch()
        let archDict = try Self.encodedArchDict(overridingArchWith: ling)

        #expect(archDict["family"] as? String == "bailingMoeV2")

        // The shared-expert width is what `ffnIntermediate` means, and it is
        // NOT the dense width, which travels in its own key. If these two ever
        // read the same, the writer has collapsed the trap the parse avoided.
        #expect(archDict["ffnIntermediate"] as? Int == 512)
        #expect(archDict["denseIntermediateSize"] as? Int == 5120)

        #expect(archDict["firstKDenseReplace"] as? Int == 1)
        #expect(archDict["numSharedExperts"] as? Int == 1)
        #expect(archDict["nGroup"] as? Int == 8)
        #expect(archDict["topkGroup"] as? Int == 4)
        #expect(archDict["routedScalingFactor"] as? Double == 2.5)
        #expect(archDict["normTopkProb"] as? Bool == true)
        #expect(archDict["scoreFunction"] as? String == "sigmoid")
        #expect(archDict["routerEnableExpertBias"] as? Bool == true)
        #expect(archDict["useQKNorm"] as? Bool == true)

        // 21 core keys + family + ten payload keys.
        #expect(archDict.count == 32)
    }

    /// `versionMinor` is hashed into `RangeCopyPlan.canonicalFingerprint`, so a
    /// bump "because the manifest schema changed" would reprice every in-flight
    /// resume checkpoint — tens of gigabytes of already-downloaded bytes —
    /// alongside reddening both frozen-layout goldens. The `family` key is
    /// additive and defaulted, so it is not a schema break.
    @Test func addingTheFamilyKeyDidNotBumpTheManifestVersion() {
        #expect(GTurboJSON.versionMajor == 1)
        #expect(GTurboJSON.versionMinor == 0)
    }

    // MARK: - Support

    /// Encode a manifest from the frozen Gemma plan, optionally swapping in a
    /// different `ArchInfo`. The writer reads nothing but `plan.arch` for the
    /// arch dict, so substituting it is enough to exercise the bailing branch
    /// without a BailingMoeV2 planner — which does not exist yet, by design.
    static func encodedArchDict(overridingArchWith arch: ArchInfo?) throws -> [String: Any] {
        let data = try GemmaFrozenPlan.withPlans { repackPlan, _ in
            let plan = arch.map { Self.plan(repackPlan, replacingArchWith: $0) } ?? repackPlan
            return try GTurboJSON.encodeManifest(
                plan: plan,
                modelID: "test/model",
                sourceSnapshotHash: String(repeating: "0", count: 64),
                files: [],
                expertsPerLayer: plan.arch.numExperts,
                numLayers: plan.arch.numLayers,
                expertStride: UInt64(getpagesize()),
                bitWidths: .init(embedding: 4, attention: 4, router: 8,
                                 sharedExpert: 4, routedExpert: 4))
        }
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let archDict = root?["arch"] as? [String: Any] else {
            Issue.record("manifest has no arch object")
            return [:]
        }
        return archDict
    }

    static func plan(_ p: RepackPlan, replacingArchWith arch: ArchInfo) -> RepackPlan {
        RepackPlan(arch: arch,
                   baseMode: p.baseMode,
                   baseGroupSize: p.baseGroupSize,
                   bitsOverrideCount: p.bitsOverrideCount,
                   resident: p.resident,
                   layers: p.layers,
                   matchedModelID: p.matchedModelID,
                   excludedMultimodalTensorNames: p.excludedMultimodalTensorNames)
    }

    static func lingArch() throws -> ArchInfo {
        let directory = GemmaFrozenPlan.temporaryRoot("bailing-manifest")
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try BailingSyntheticConfig.write(to: directory)
        return try ArchInfo.load(configPath: path)
    }
}
