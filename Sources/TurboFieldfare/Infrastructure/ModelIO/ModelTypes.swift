import Foundation
import Metal

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String
    /// Which architecture this describes, carrying the facts only that family
    /// has. Everything above is a field both families genuinely populate; a
    /// fact that exists for one family only belongs in the payload, never as a
    /// fabricated value on a core field.
    public let variant: ArchVariant

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String,
        // Deliberately has no default. A defaulted `.gemma4` would let a
        // future architecture's baseline compile while silently claiming to be
        // Gemma, and `validateArch` would then compare Gemma's fields against
        // values the other family never had.
        variant: ArchVariant
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.variant = variant
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh",
        variant: .gemma4
    )

    /// `inclusionAI/Ling-mini-2.0`, consumed as `mlx-community/Ling-mini-2.0-4bit`.
    ///
    /// **Deliberately absent from `supported`.** This build ships no BailingMoeV2
    /// kernels, and `supported` means "this build can execute these bytes": an
    /// entry here without kernels lets the model install cleanly and then trap at
    /// the first attention dispatch. It is added alongside the kernels, not before.
    /// `ArchConfigLingTests.aRealLingInstallCannotLoadWithoutItsKernels` is the
    /// guard: it asserts the consequence — the load is refused — rather than
    /// which list the name appears in.
    ///
    /// `intermediateSize = 512` is the SHARED-EXPERT FFN width
    /// (`moe_shared_expert_intermediate_size`), which is what this field means and
    /// how the runtime consumes it. Ling's config also has a key literally named
    /// `intermediate_size` = 5120, the dense layer-0 FFN width; that lives in
    /// `BailingMoeV2Extras.denseIntermediateSize`. Putting 5120 here would silently
    /// size the shared-expert GEMV ten times wrong.
    ///
    /// Ling draws no sliding/global distinction, so `numFullKVHeads`/`fullHeadDim`
    /// mirror the head geometry and `slidingWindow`/`finalLogitSoftcap` are 0,
    /// meaning "none". These are definite values, not placeholders:
    /// `validateArch` compares all six for Ling too, and `numFullKVHeads` /
    /// `fullHeadDim` are exactly the geometry a 100%-full-attention runtime path
    /// reads, so an unchecked one would trap at dispatch rather than at load.
    public static let lingMini2_0_4bit = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 4,
        numFullKVHeads: 4,
        headDim: 128,
        fullHeadDim: 128,
        vocabSize: 157184,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 600_000.0,
        fullRopeTheta: 600_000.0,
        partialRotaryFactor: 0.5,
        numLayers: 20,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        // Ling's K and V are distinct projections. The flag carries no safety on
        // its own — the Gemma runner derives V from K for any full-attention
        // layer without consulting it — which is why Ling needs its own forward
        // path rather than a `false` here.
        attentionKEqV: false,
        // No sliding window and no `layer_types`: every layer is full attention.
        fullAttentionLayerMask: [UInt8](repeating: 1, count: 20),
        hiddenActivation: "silu",
        variant: .bailingMoeV2(BailingMoeV2Extras(
            denseIntermediateSize: 5120,
            firstKDenseReplace: 1,
            numSharedExperts: 1,
            nGroup: 8,
            topkGroup: 4,
            routedScalingFactor: 2.5,
            normTopkProb: true,
            scoreFunction: "sigmoid",
            routerEnableExpertBias: true,
            useQKNorm: true))
    )

    /// Every architecture this build can execute. An entry here without the
    /// matching Metal kernels would let a model install and then trap at the
    /// first attention dispatch, so entries are added only alongside kernels.
    /// `lingMini2_0_4bit` is defined above and stays out for exactly that reason.
    public static let supported: [ArchConfig] = [gemma4_26B_A4B]

    /// The families this build has kernels for — derived from `supported`, never
    /// listed by hand, so the two cannot disagree.
    ///
    /// **This, not `real`, is what the executability gate is keyed on.** Keying
    /// it on membership in `real` refuses only the exact baselines this package
    /// happens to have transcribed: a *different* BailingMoeV2 checkpoint —
    /// another Ling size, a fine-tune with different dimensions — matches no
    /// entry in `real`, so that gate never fires and the manifest is refused
    /// only if some caller's baseline happens to disagree with it. What has to
    /// hold is architectural: there are no BailingMoeV2 kernels in this binary,
    /// so no BailingMoeV2 manifest is executable, whatever its dimensions.
    public static let executableFamilies: Set<ArchFamily> =
        Set(supported.map { $0.variant.family })

    /// Every published checkpoint this package has a real baseline for, whether
    /// or not this build ships kernels for it. `supported` is a subset.
    ///
    /// This is the list that decides whether `manifest.quant` is mandatory. A
    /// real checkpoint's weights are quantized, so a manifest matching one of
    /// these and carrying no `quant` block is incomplete — and without this
    /// list, a real Ling install could skip the per-architecture bit-width table
    /// entirely just by omitting the block, because Ling is deliberately not in
    /// `supported` yet. Every entry needs its own `acceptedQuantBits` row.
    public static let real: [ArchConfig] = [gemma4_26B_A4B, lingMini2_0_4bit]

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }
}

extension ManifestArch {
    /// The manifest shape a given baseline would produce. Copies every field
    /// `ManifestReader.validateArch` compares, so a round trip through this
    /// initialiser is by construction a match.
    public init(from config: ArchConfig) {
        self.init(
            hiddenSize: config.hiddenSize,
            ffnIntermediate: config.intermediateSize,
            moeIntermediateSize: config.moeIntermediateSize,
            numHeads: config.numHeads,
            numKVHeads: config.numKVHeads,
            numFullKVHeads: config.numFullKVHeads,
            headDim: config.headDim,
            fullHeadDim: config.fullHeadDim,
            vocabSize: config.vocabSize,
            slidingWindow: config.slidingWindow,
            finalLogitSoftcap: config.finalLogitSoftcap,
            ropeTheta: config.ropeTheta,
            fullRopeTheta: config.fullRopeTheta,
            partialRotaryFactor: config.partialRotaryFactor,
            numLayers: config.numLayers,
            numExperts: config.numExperts,
            topKExperts: config.topKExperts,
            tieWordEmbeddings: config.tieWordEmbeddings,
            attentionKEqV: config.attentionKEqV,
            hiddenActivation: config.hiddenActivation,
            fullAttentionLayerMask: config.fullAttentionLayerMask.map { Int($0) },
            variant: config.variant)
    }
}

/// Failure modes for the validation gates in `Model.load`.
enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    /// The manifest describes an architecture FAMILY this build has no kernels
    /// for — any manifest of that family, not only the published checkpoint
    /// this package happens to know the dimensions of. Distinct from
    /// `archMismatch`: nothing is wrong with the file, and no other build would
    /// reject it — this one simply cannot execute it.
    case architectureNotExecutable(family: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .architectureNotExecutable(let family):
            return "this build has no kernels for architecture \(family)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
