// The declarations in this file exist twice, once per target:
//   Sources/TurboFieldfare/Infrastructure/ModelIO/ArchFamily.swift  (beside ArchConfig)
//   Sources/TurboFieldfareRepack/Core/Format/ArchFamily.swift       (beside ArchInfo)
// The two files are identical apart from access level. That is deliberate.
//
// `Package.swift` gives `TurboFieldfare` and `TurboFieldfareRepackCore` no
// dependency edge in either direction — the repacker writes `manifest.json`,
// the runtime reads it, and the file format is the only contract between them.
// Sharing them would need a `TurboFieldfareRepackCore -> TurboFieldfare` edge,
// which drags swift-transformers and the whole Metal runtime into a
// command-line repacker that needs neither — for the sake of three
// declarations that change once per architecture. A third shared module would
// instead cost a new target in every dependency list. Note the RepackCore copy
// is `internal`, so it cannot make a bare `ArchFamily` ambiguous anywhere; the
// cost of sharing is dependency weight, not name resolution.
//
// The copies are held together by `ArchFamilyCrossTargetTests`, which compares
// the case list and the stored-property names structurally, because `==` cannot
// cross a module boundary. Edit one copy and that test fails.

/// The model architectures this package can describe. The family is the
/// discriminator carried in `manifest.json -> arch -> family`; it rides in
/// `arch` rather than `flags` because the manifest reader rejects unknown
/// `flags` keys outright.
enum ArchFamily: String, Sendable, Equatable, CaseIterable, Codable {
    case gemma4
    case bailingMoeV2
}

/// BailingMoeV2 facts that have no core arch field, because Gemma has no
/// equivalent concept.
///
/// Note what is *not* here: `intermediateSize` stays a core field and holds the
/// **shared-expert** FFN width (512 for Ling), which is how the runtime already
/// consumes it. Ling's `config.json` also has a key literally named
/// `intermediate_size` (5120) meaning the dense layer-0 FFN — a different
/// quantity that lives in `denseIntermediateSize` below.
struct BailingMoeV2Extras: Sendable, Equatable, Codable {
    /// Dense layer-0 FFN width — `config.json -> intermediate_size`. Not the
    /// shared-expert width; see the note above.
    let denseIntermediateSize: Int
    /// Number of leading layers that are dense: no router, no routed experts.
    let firstKDenseReplace: Int
    let numSharedExperts: Int
    /// Expert groups for group-limited routing, and how many groups survive.
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Double
    let normTopkProb: Bool
    /// Router score activation, e.g. "sigmoid" — Gemma's router is softmax.
    let scoreFunction: String
    /// Whether a learned per-expert bias shifts the router scores. The bias
    /// affects *selection* only; the returned weight is the unbiased score.
    let routerEnableExpertBias: Bool
    /// Per-head query/key layernorm. There is no V norm.
    let useQKNorm: Bool

    init(
        denseIntermediateSize: Int,
        firstKDenseReplace: Int,
        numSharedExperts: Int,
        nGroup: Int,
        topkGroup: Int,
        routedScalingFactor: Double,
        normTopkProb: Bool,
        scoreFunction: String,
        routerEnableExpertBias: Bool,
        useQKNorm: Bool
    ) {
        self.denseIntermediateSize = denseIntermediateSize
        self.firstKDenseReplace = firstKDenseReplace
        self.numSharedExperts = numSharedExperts
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.routedScalingFactor = routedScalingFactor
        self.normTopkProb = normTopkProb
        self.scoreFunction = scoreFunction
        self.routerEnableExpertBias = routerEnableExpertBias
        self.useQKNorm = useQKNorm
    }
}

/// A family together with the facts only that family has. Matching on a variant
/// pair is what lets validation compare each family's own payload while the
/// fields both families genuinely carry stay core and are compared for both.
enum ArchVariant: Sendable, Equatable {
    /// No payload: every Gemma fact is already a core field of the arch
    /// description (`ArchConfig` / `ArchInfo`) that the Metal-facing runtime
    /// reads directly, and a second copy of a field the runtime already reads
    /// is a second source of truth that validation and execution can disagree
    /// about. An empty payload struct here would be a placeholder for nothing;
    /// when a genuinely Gemma-only fact appears, it gets one then.
    case gemma4
    case bailingMoeV2(BailingMoeV2Extras)

    var family: ArchFamily {
        switch self {
        case .gemma4: .gemma4
        case .bailingMoeV2: .bailingMoeV2
        }
    }
}
