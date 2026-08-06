import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareRepackCore

/// `ArchFamily`, `ArchVariant` and `BailingMoeV2Extras` are
/// declared once in `TurboFieldfare` (beside `ArchConfig`) and again in
/// `TurboFieldfareRepackCore` (beside `ArchInfo`), because the two targets have
/// no dependency edge — see the comment at the top of either `ArchFamily.swift`.
///
/// The repacker writes the manifest and the runtime validates it, so the two
/// copies drifting apart means a model that repacks and then refuses to load,
/// with the error naming a field rather than the drift. This suite is the only
/// thing holding them together: it is the test the header comments promise.
///
/// The comparison is structural, not `==`: two identically-named types from two
/// modules are two distinct types, so the compiler will not compare them and
/// `Mirror` label lists are the closest available proxy. Every reference is
/// module-qualified, because inside this test target both copies are visible
/// and a bare `ArchFamily` is ambiguous.
@Suite struct ArchFamilyCrossTargetTests {

    /// Label plus the printed value of every stored property, in declaration
    /// order. Including the value catches a field whose *type* changed (an `Int`
    /// prints `2`, a `Double` prints `2.0`) — labels alone would not.
    private static func fields(_ subject: Any) -> [String] {
        Mirror(reflecting: subject).children.map { child in
            "\(child.label ?? "<unlabelled>")=\(String(describing: child.value))"
        }
    }

    @Test func familyCaseListsAgreeAcrossTargets() {
        let runtime = TurboFieldfare.ArchFamily.allCases.map(\.rawValue)
        let repack = TurboFieldfareRepackCore.ArchFamily.allCases.map(\.rawValue)

        #expect(runtime == repack)

        // Absolute pin as well as a relative one: the comparison above still
        // passes if both copies gain the same third family, and the raw values
        // are what a shipped `manifest.json` carries, so renaming one silently
        // orphans every manifest already on disk.
        #expect(runtime == ["gemma4", "bailingMoeV2"])
    }

    /// `.gemma4` carries no payload on either side. Pinned because adding one to
    /// a single copy compiles fine within that target and only shows up as a
    /// manifest that repacks and then refuses to load.
    @Test func gemma4CaseCarriesNoPayloadOnEitherTarget() {
        #expect(Self.fields(TurboFieldfare.ArchVariant.gemma4) == [])
        #expect(Self.fields(TurboFieldfareRepackCore.ArchVariant.gemma4) == [])
    }

    @Test func bailingMoeV2ExtrasAgreesAcrossTargets() {
        // Deliberately not Ling's real values: distinct numbers per field so a
        // transposed pair of same-typed properties shows up as a mismatch.
        let runtime = TurboFieldfare.BailingMoeV2Extras(
            denseIntermediateSize: 5120,
            firstKDenseReplace: 11,
            numSharedExperts: 12,
            nGroup: 13,
            topkGroup: 14,
            routedScalingFactor: 2.5,
            normTopkProb: true,
            scoreFunction: "sigmoid",
            routerEnableExpertBias: false,
            useQKNorm: true)
        let repack = TurboFieldfareRepackCore.BailingMoeV2Extras(
            denseIntermediateSize: 5120,
            firstKDenseReplace: 11,
            numSharedExperts: 12,
            nGroup: 13,
            topkGroup: 14,
            routedScalingFactor: 2.5,
            normTopkProb: true,
            scoreFunction: "sigmoid",
            routerEnableExpertBias: false,
            useQKNorm: true)

        #expect(Self.fields(runtime) == Self.fields(repack))

        // Absolute pin: both copies gaining the same field, or losing one, is
        // exactly as breaking for the manifest contract as one copy drifting.
        #expect(Mirror(reflecting: runtime).children.compactMap(\.label) == [
            "denseIntermediateSize",
            "firstKDenseReplace",
            "numSharedExperts",
            "nGroup",
            "topkGroup",
            "routedScalingFactor",
            "normTopkProb",
            "scoreFunction",
            "routerEnableExpertBias",
            "useQKNorm",
        ], "BailingMoeV2Extras gained or lost a field; mirror it in the other target")
    }

    @Test func variantReportsItsFamilyOnBothTargets() {
        #expect(TurboFieldfare.ArchVariant.gemma4.family == .gemma4)
        #expect(TurboFieldfareRepackCore.ArchVariant.gemma4.family == .gemma4)

        let runtimeBailing = TurboFieldfare.ArchVariant.bailingMoeV2(
            .init(denseIntermediateSize: 5120, firstKDenseReplace: 1,
                  numSharedExperts: 1, nGroup: 8, topkGroup: 4,
                  routedScalingFactor: 2.5, normTopkProb: true,
                  scoreFunction: "sigmoid", routerEnableExpertBias: true,
                  useQKNorm: true))
        let repackBailing = TurboFieldfareRepackCore.ArchVariant.bailingMoeV2(
            .init(denseIntermediateSize: 5120, firstKDenseReplace: 1,
                  numSharedExperts: 1, nGroup: 8, topkGroup: 4,
                  routedScalingFactor: 2.5, normTopkProb: true,
                  scoreFunction: "sigmoid", routerEnableExpertBias: true,
                  useQKNorm: true))

        #expect(runtimeBailing.family == .bailingMoeV2)
        #expect(repackBailing.family == .bailingMoeV2)
    }
}
