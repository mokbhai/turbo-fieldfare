import Foundation

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

/// Fields are `var` so a candidate architecture can be built and perturbed
/// without decoding a manifest; see `ManifestArch.init(from:)`.
public struct ManifestArch: Decodable, Equatable, Sendable {
    public var hiddenSize: Int
    public var ffnIntermediate: Int
    public var moeIntermediateSize: Int
    public var numHeads: Int
    public var numKVHeads: Int
    public var numFullKVHeads: Int
    public var headDim: Int
    public var fullHeadDim: Int
    public var vocabSize: Int
    public var slidingWindow: Int
    public var finalLogitSoftcap: Double
    public var ropeTheta: Double
    public var fullRopeTheta: Double
    public var partialRotaryFactor: Double
    public var numLayers: Int
    public var numExperts: Int
    public var topKExperts: Int
    public var tieWordEmbeddings: Bool
    public var attentionKEqV: Bool
    public var hiddenActivation: String
    public var fullAttentionLayerMask: [Int]
    /// Which family wrote this manifest, plus that family's own facts. Defaults
    /// to `.gemma4` when `arch.family` is absent — see `init(from:)`.
    public var variant: ArchVariant
}

// This extension is deliberately not in the struct body: an initializer
// declared there would suppress the memberwise init that
// `ManifestArch.init(from config:)` calls and `ManifestArchListTests` perturbs.
extension ManifestArch {
    /// The wire format. These names are the property names verbatim, which is
    /// what the previous synthesized `.useDefaultKeys` decoding used and what
    /// `GTurboJSON.archDict` writes. Renaming one silently stops reading a
    /// manifest that is already on a user's disk.
    private enum CodingKeys: String, CodingKey {
        case hiddenSize, ffnIntermediate, moeIntermediateSize
        case numHeads, numKVHeads, numFullKVHeads
        case headDim, fullHeadDim, vocabSize
        case slidingWindow, finalLogitSoftcap
        case ropeTheta, fullRopeTheta, partialRotaryFactor
        case numLayers, numExperts, topKExperts
        case tieWordEmbeddings, attentionKEqV
        case hiddenActivation, fullAttentionLayerMask
        case family
    }

    /// Decoded by hand, not by synthesis, for one reason: **every manifest
    /// shipped so far predates `family` and must keep decoding exactly as it
    /// did.** `family` is therefore the only optional key, defaulting to
    /// `.gemma4`.
    ///
    /// Every other key uses `decode`, never `decodeIfPresent` with a default: a
    /// truncated manifest must surface as `keyNotFound` — wrapped by
    /// `ManifestReader.load` into `indexCorrupt` — rather than quietly acquiring
    /// a plausible value for a field nobody supplied.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        ffnIntermediate = try c.decode(Int.self, forKey: .ffnIntermediate)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        numKVHeads = try c.decode(Int.self, forKey: .numKVHeads)
        numFullKVHeads = try c.decode(Int.self, forKey: .numFullKVHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        fullHeadDim = try c.decode(Int.self, forKey: .fullHeadDim)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        finalLogitSoftcap = try c.decode(Double.self, forKey: .finalLogitSoftcap)
        ropeTheta = try c.decode(Double.self, forKey: .ropeTheta)
        fullRopeTheta = try c.decode(Double.self, forKey: .fullRopeTheta)
        partialRotaryFactor = try c.decode(Double.self, forKey: .partialRotaryFactor)
        numLayers = try c.decode(Int.self, forKey: .numLayers)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        topKExperts = try c.decode(Int.self, forKey: .topKExperts)
        tieWordEmbeddings = try c.decode(Bool.self, forKey: .tieWordEmbeddings)
        attentionKEqV = try c.decode(Bool.self, forKey: .attentionKEqV)
        hiddenActivation = try c.decode(String.self, forKey: .hiddenActivation)
        fullAttentionLayerMask = try c.decode([Int].self, forKey: .fullAttentionLayerMask)

        // `contains`, not `decodeIfPresent`: the latter cannot tell an absent
        // key from an explicit `"family": null`, and the two mean opposite
        // things. Absent means "written before `family` existed" and must keep
        // defaulting to gemma4, because every shipped manifest is shaped that
        // way and its bytes are hashed into `VerifiedInstallReceipt`. Present
        // and null means a writer emitted a family it could not name, which is
        // malformed — `decode` turns it into `valueNotFound`.
        let family: ArchFamily
        if c.contains(.family) {
            family = try c.decode(ArchFamily.self, forKey: .family)
        } else {
            family = .gemma4
        }

        // The payload keys are flat inside `arch`, so the variant decodes from
        // the same decoder. Modelling them as optional properties here instead
        // would make a bailing manifest that omits `denseIntermediateSize`
        // decode successfully as `nil` and then compare equal to another `nil`.
        switch family {
        case .gemma4:
            variant = .gemma4
        case .bailingMoeV2:
            variant = .bailingMoeV2(try BailingMoeV2Extras(from: decoder))
        }
    }
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = [
        "streamingPresent", "turboQuantKV", "aneSharedExpert"
    ]

    /// Required file entries (relative to `model.gturbo/`). Layer files
    /// `packed_experts/layer_<L>.bin` for L in 0..<numLayers are checked
    /// after decode against `numLayers` (with the zero-padded "layer_%02d"
    /// naming the writer produces; falling back to plain "layer_<L>" when
    /// only the unpadded form is present, for toy synthetics).
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    /// Read and validate `manifest.json`.
    ///
    /// **`expecting:` is a fallback baseline, not a demand.** A manifest that
    /// matches any entry of `ArchConfig.supported` is accepted whatever the
    /// caller passed, because `supported` is the set this build ships kernels
    /// for and executability is the property that actually matters; the
    /// caller's baseline is then only used to *report* why a non-matching
    /// manifest was refused, so a near-miss still names the field that
    /// differs. `manifestMatchingSupportedLoadsAgainstAnyBaseline` pins that.
    ///
    /// The consequence worth stating out loud: `expecting:` cannot be used to
    /// let a model in. A manifest whose *family* this build has no kernels for
    /// is refused whatever baseline is passed, including that architecture's
    /// own and including a toy baseline invented by a test — see the
    /// `architectureNotExecutable` gate at the end of `validate`. A toy
    /// baseline of an *executable* family stays admissible, which is how the
    /// tests exercise validation without pretending to execute anything.
    public static func load(directoryURL: URL,
                            expecting: ArchConfig,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let size = try metadataFileSize(manifestURL, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "manifest.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting,
                     directoryURL: directoryURL)
        return manifest
    }

    private static func metadataFileSize(_ url: URL,
                                         fileName: String) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attrs[.size] as? NSNumber else {
            throw ModelError.indexCorrupt(detail: "\(fileName): file size unavailable")
        }
        return number.uint64Value
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig,
                         directoryURL: URL) throws {
        guard m.magic == "GTURBO" else { throw ModelError.notAGTurboDirectory }
        guard m.versionMajor == 1 else {
            throw ModelError.unsupportedVersion(major: m.versionMajor, minor: m.versionMinor)
        }
        for key in m.flags.keys {
            if !knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
        }
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        // Any architecture this build ships kernels for is acceptable; the
        // caller's `expected` baseline is the reason-reporting path, so a
        // near-miss still names the field that differs. See `load` for what
        // that makes `expecting:` mean.
        //
        // `supported`, not `real`: widening this to `real` would let a Ling
        // manifest skip `validateArch` entirely, so the refusal at the end of
        // this function would report "no kernels" for a manifest that is also
        // the wrong architecture for the caller — the less useful of the two
        // true statements. `aRealLingInstallCannotLoadWithoutItsKernels` pins
        // that a Gemma-baselined caller is told `family` first.
        //
        // One call, not two. The executability gate at the end of this function
        // used to recompute this same expression, and `matchArch` runs the whole
        // field-by-field `validateArch` against every candidate — so it ran
        // twice for no reason. The gate now asks a different question entirely
        // (the family), so nothing downstream needs this value.
        if Self.matchArch(m.arch, against: ArchConfig.supported) == nil {
            try validateArch(m.arch, expected: expected)
        }
        // The manifest decides whether quant metadata is mandatory, not a guess
        // about the caller's baseline. Matching a `real` entry means these bytes
        // describe a published checkpoint, and a published checkpoint's weights
        // are quantized — so metadata describing that quantization is not
        // optional. The previous numLayers/hiddenSize comparison against the
        // Gemma baseline was a stand-in for that and would have started
        // answering for other architectures by coincidence.
        //
        // Deliberately keyed off `real` and not `supported`: Ling is out of
        // `supported` until its kernels land, and keying off `supported` would
        // let a real Ling install skip the per-architecture bit-width table
        // altogether just by omitting the block. Since `supported` is a subset
        // of `real`, every manifest that needed a quant block before still does.
        //
        // `matchedReal ?? expected` matters: the bit-width table is
        // per-architecture, so a toy baseline still needs its own row rather
        // than being validated against Gemma's.
        let matchedReal = Self.matchArch(m.arch, against: ArchConfig.real)
        if let quant = m.quant {
            try validateQuant(quant, expecting: matchedReal ?? expected)
        } else if matchedReal != nil {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required for architecture "
                      + m.arch.variant.family.rawValue)
        }
        // The top-level `numLayers` is a separate number from `arch.numLayers`
        // and nothing above compares it, so it arrives from the file
        // unconstrained. `0..<m.numLayers` below is a TRAP for a negative
        // bound, not a throw — a manifest saying `"numLayers": -1` aborted the
        // process rather than being reported as corrupt.
        guard m.numLayers >= 0 else {
            throw ModelError.indexCorrupt(
                detail: "manifest.numLayers is negative: \(m.numLayers)")
        }
        let pageSize = UInt64(getpagesize())
        guard m.expertStride % pageSize == 0 else {
            throw ModelError.expertStrideNotPageAligned(stride: m.expertStride,
                                                        pageSize: Int(pageSize))
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
        for L in 0..<m.numLayers {
            let padded = String(format: "packed_experts/layer_%02d.bin", L)
            let plain  = "packed_experts/layer_\(L).bin"
            if m.files[padded] == nil && m.files[plain] == nil {
                throw ModelError.missingFile(name: padded)
            }
        }
        // **The milestone's safety property, unconditional.** A manifest whose
        // family this build has no kernels for cannot be loaded, whatever its
        // dimensions and whatever baseline the caller passed. Nothing above
        // refuses it on its own: the `expected` baseline is only a fallback
        // (see `load`), so a caller that passed that architecture's own
        // baseline would validate it clean and then trap at the first attention
        // dispatch, with nothing pointing back at the cause. This makes
        // "kernels land before the architecture becomes loadable" a gate rather
        // than a convention.
        //
        // **Keyed on the FAMILY, not on membership in `real`.** `real` is the
        // list of checkpoints this package has transcribed a baseline for, and
        // an unrecognised BailingMoeV2 manifest — another Ling size, a
        // fine-tune, a corrupted copy of the one we know — matches nothing in
        // it. Keying the gate there refused exactly one file and let every
        // other file of the same unexecutable architecture through, and the
        // property held only because every in-package caller passes the Gemma
        // baseline and the family check inside `validateArch` catches it first.
        // Executability is a property of the architecture, not of one set of
        // dimensions, so the gate asks the architectural question. It also
        // generalises: the next family gets this for free, before anyone
        // remembers to add its baseline to `real`.
        //
        // Note this can no longer be satisfied by `matchedSupported != nil`
        // alone. A manifest of an executable family that matches no `supported`
        // entry is a *different* verdict — it was already reported field-by-
        // field by `validateArch` above — and reaching here means it matched
        // the caller's baseline, which is the documented meaning of `expecting:`.
        //
        // Last, deliberately. Everything before it reports what is wrong with
        // the FILE, and those verdicts must not depend on which build is
        // reading — a truncated or mis-quantized manifest gets the same error
        // here as it will once the kernels exist. This is the only verdict
        // that is about this build, so it is the only one that may change.
        let family = m.arch.variant.family
        if !ArchConfig.executableFamilies.contains(family) {
            throw ModelError.architectureNotExecutable(family: family.rawValue)
        }
    }

    private static func validateQuant(_ quant: ManifestQuant,
                                      expecting: ArchConfig) throws {
        let bits = expecting.variant.acceptedQuantBits
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, bits.embedding),
            ("attention", quant.attention, bits.attention),
            ("router", quant.router, bits.router),
            ("sharedExpert", quant.sharedExpert, bits.sharedExpert),
            ("routedExpert", quant.routedExpert, bits.routedExpert),
        ]
        for (name, slot, allowedBits) in slots {
            guard allowedBits.contains(slot.weightBits),
                  slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16",
                  slot.groupSize == Quantization.groupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
    }

    /// Returns the first supported architecture that matches the manifest, or
    /// nil. Callers that need a field-level reason fall through to
    /// `validateArch`, which throws `archMismatch` naming the first difference.
    static func matchArch(_ manifestArch: ManifestArch,
                          against candidates: [ArchConfig]) -> ArchConfig? {
        candidates.first { candidate in
            (try? validateArch(manifestArch, expected: candidate)) != nil
        }
    }

    /// Compares the family, then all 21 core fields, then only the payload
    /// belonging to the matched variant.
    ///
    /// The family gate is first so that a manifest from the wrong architecture
    /// reports `family` rather than whichever core field happens to differ
    /// first — "hiddenSize 2048, expected 2816" sends the reader looking for a
    /// corrupt Gemma install when what they have is a Ling one. Every thrown
    /// `field:` string after it, and their order, are unchanged from before this
    /// became variant-aware.
    ///
    /// **All 21 core fields are compared for both families**, including
    /// `numFullKVHeads`, `fullHeadDim`, `slidingWindow`, `finalLogitSoftcap`,
    /// `ropeTheta` and `attentionKEqV`, which read as Gemma concepts but are
    /// not: `ManifestArch` decodes every one of them mandatorily, the writer
    /// emits every one of them, and Ling holds a definite value for each. Two of
    /// them matter most for Ling precisely because it is 100% full attention —
    /// `numFullKVHeads` and `fullHeadDim` are the geometry its runtime path
    /// reads, so leaving them unchecked would let a corrupted Ling manifest pass
    /// validation and trap at dispatch instead.
    ///
    /// A field is only payload when the *other* family has no value for it at
    /// all. `slidingWindow: 0` and `finalLogitSoftcap: 0.0` on Ling are that
    /// family's real, definite "none", not fabrications — so they are compared
    /// like everything else, and a manifest claiming otherwise is corrupt.
    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("family",
                  a.variant.family.rawValue,
                  e.variant.family.rawValue)

        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        // Compared in `Int`, and in this direction, because everything on the
        // `a` side came out of a file nobody in this process wrote. The mask
        // used to be narrowed with `a.fullAttentionLayerMask.map { UInt8($0) }`,
        // and `UInt8.init(_:)` TRAPS: a manifest carrying `-1` or `999` aborted
        // the process instead of being rejected. `Int.init(_ UInt8)` is total,
        // so widening the baseline instead cannot fail, and an out-of-range
        // element is reported as the `archMismatch` it is. The printed forms of
        // `[Int]` and `[UInt8]` are identical, so no existing message moves.
        try check("fullAttentionLayerMask",
                  a.fullAttentionLayerMask.description,
                  e.fullAttentionLayerMask.map { Int($0) }.description)

        switch (a.variant, e.variant) {
        case (.gemma4, .gemma4):
            // Nothing to compare: Gemma has no fact that is not already one of
            // the core fields above, which is why `.gemma4` carries no payload.
            break
        case let (.bailingMoeV2(av), .bailingMoeV2(ev)):
            try check("denseIntermediateSize",
                      av.denseIntermediateSize, ev.denseIntermediateSize)
            try check("firstKDenseReplace",
                      av.firstKDenseReplace, ev.firstKDenseReplace)
            try check("numSharedExperts",  av.numSharedExperts,  ev.numSharedExperts)
            try check("nGroup",            av.nGroup,            ev.nGroup)
            try check("topkGroup",         av.topkGroup,         ev.topkGroup)
            try check("routedScalingFactor",
                      av.routedScalingFactor, ev.routedScalingFactor)
            try check("normTopkProb",      av.normTopkProb,      ev.normTopkProb)
            try check("scoreFunction",     av.scoreFunction,     ev.scoreFunction)
            try check("routerEnableExpertBias",
                      av.routerEnableExpertBias, ev.routerEnableExpertBias)
            try check("useQKNorm",         av.useQKNorm,         ev.useQKNorm)
        case (.gemma4, .bailingMoeV2), (.bailingMoeV2, .gemma4):
            // Unreachable today: the family gate at the top of this function
            // already rejected a mismatched pair, and every case here is
            // spelled out rather than left to a `default:` precisely so it
            // stays that way. A third family makes this switch non-exhaustive
            // and the compiler names the pairs nobody decided about, where a
            // `default:` would have quietly swallowed them and compared no
            // payload at all.
            throw ModelError.archMismatch(field: "family",
                                          expected: e.variant.family.rawValue,
                                          actual: a.variant.family.rawValue)
        }
    }
}

/// Accepted weight bit widths for each quantized slot. Per-architecture, not
/// package-wide: the values encode which Metal kernels exist, and those differ
/// by family.
struct AcceptedQuantBits {
    let embedding: Set<Int>
    let attention: Set<Int>
    let router: Set<Int>
    let sharedExpert: Set<Int>
    let routedExpert: Set<Int>
}

// Lives here rather than in `ArchFamily.swift` so that file stays literally
// identical to its `TurboFieldfareRepackCore` twin; the repacker has no use for
// a table describing which kernels the runtime ships.
extension ArchVariant {
    /// Bit widths the kernels can actually read, per slot.
    ///
    /// The **router** row is the one that matters. Gemma's router kernels
    /// (`router_gemv_gemma4_body`, `prefill_router_gemma4_block`) take
    /// `device const uint8_t*` and index one byte per weight; handed a genuinely
    /// 4-bit blob they read twice its length and produce garbage logits with no
    /// assertion anywhere. Ling's router is 4-bit. Widening a shared table to
    /// `[4, 8]` to accommodate both would silently admit that install — which is
    /// why the table is keyed by architecture instead.
    ///
    /// Gemma's `sharedExpert: [4, 8]` is a real allowance for historical 8-bit
    /// installs, not a shrug.
    var acceptedQuantBits: AcceptedQuantBits {
        switch self {
        case .gemma4:
            AcceptedQuantBits(embedding: [4],
                              attention: [4],
                              router: [8],
                              sharedExpert: [4, 8],
                              routedExpert: [4])
        case .bailingMoeV2:
            AcceptedQuantBits(embedding: [4],
                              attention: [4],
                              router: [4],
                              sharedExpert: [4],
                              routedExpert: [4])
        }
    }
}
