import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct ManifestReaderTests {

    /// Build a manifest dictionary for a 2-layer toy ArchConfig and write it
    /// into a temp directory. Returns the directory URL and the toy config.
    static func writeToyManifest(_ overrides: [String: Any] = [:],
                                 flags: [String: Bool] = ["streamingPresent": true,
                                                          "turboQuantKV": false,
                                                          "aneSharedExpert": false],
                                 archOverrides: [String: Any] = [:],
                                 archRemovals: [String] = [],
                                 filesOverride: [String: [String: Any]]? = nil,
                                 config: ArchConfig = .gemma4Toy()) throws
                                 -> (URL, ArchConfig) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let toy = config
        var archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize,
            "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads,
            "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim,
            "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize,
            "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta,
            "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers,
            "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        // Mirrors `GTurboJSON.encodeManifest`: no `family` key for Gemma, so
        // every Gemma case here keeps proving the reader's `?? .gemma4` default
        // against a manifest shaped exactly like the shipped ones.
        if case .bailingMoeV2(let extras) = toy.variant {
            archDict["family"] = "bailingMoeV2"
            archDict["denseIntermediateSize"] = extras.denseIntermediateSize
            archDict["firstKDenseReplace"] = extras.firstKDenseReplace
            archDict["numSharedExperts"] = extras.numSharedExperts
            archDict["nGroup"] = extras.nGroup
            archDict["topkGroup"] = extras.topkGroup
            archDict["routedScalingFactor"] = extras.routedScalingFactor
            archDict["normTopkProb"] = extras.normTopkProb
            archDict["scoreFunction"] = extras.scoreFunction
            archDict["routerEnableExpertBias"] = extras.routerEnableExpertBias
            archDict["useQKNorm"] = extras.useQKNorm
        }
        for (k, v) in archOverrides { archDict[k] = v }
        for k in archRemovals { archDict.removeValue(forKey: k) }

        var files: [String: [String: Any]]
        if let f = filesOverride {
            files = f
        } else {
            files = [
                "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            ]
            for L in 0..<toy.numLayers {
                files["packed_experts/layer_\(L).bin"] = ["size": 16384, "sha256": String(repeating: "0", count: 64)]
            }
        }

        var root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": flags,
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": 16384,
        ]
        for (k, v) in overrides { root[k] = v }

        let data = try JSONSerialization.data(withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return (dir, toy)
    }

    /// The `quant` block a real install of `family` carries, optionally with one
    /// field of one slot replaced.
    ///
    /// The five widths are **literals per family**, not reads of
    /// `ArchVariant.acceptedQuantBits`. A fixture derived from the table under
    /// test would move with it: widen Gemma's router to `[4, 8]` and a
    /// generated fixture would start writing 4, so the test asserting 4 is
    /// refused would be handed a manifest it no longer objects to and could
    /// pass by agreeing with the regression.
    ///
    /// One replaced field, not a set of per-slot knobs, because every case that
    /// uses this asks the same question: with the other four slots correct,
    /// does *this* one still bite?
    static func quantBlock(_ family: ArchFamily,
                           slot: String? = nil,
                           field: String = "weightBits",
                           value: Any? = nil) -> [String: Any] {
        let widths: [String: Int]
        switch family {
        case .gemma4:
            // 4-bit throughout except the router, whose kernels index one byte
            // per weight.
            widths = ["embedding": 4, "attention": 4, "router": 8,
                      "sharedExpert": 4, "routedExpert": 4]
        case .bailingMoeV2:
            // `mlx-community/Ling-mini-2.0-4bit` — 4-bit throughout, router
            // included.
            widths = ["embedding": 4, "attention": 4, "router": 4,
                      "sharedExpert": 4, "routedExpert": 4]
        }
        var block: [String: Any] = [:]
        for (name, bits) in widths {
            var s: [String: Any] = [
                "weightBits": bits,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": Quantization.groupSize,
            ]
            if name == slot, let value { s[field] = value }
            block[name] = s
        }
        return block
    }

    @Test func loadsValidManifest() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.magic == "GTURBO")
        #expect(m.numLayers == toy.numLayers)
        #expect(m.expertStride == 16384)
    }

    @Test func missingManifestThrowsPartialInstall() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .gemma4Toy())
        } throws: { error in
            if case ModelError.partialInstall = error { return true }
            return false
        }
    }

    @Test func oversizedManifestRejectsBeforeDecode() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try Data(repeating: 0x20, count: 64).write(to: manifestURL)

        #expect {
            _ = try ManifestReader.load(directoryURL: dir,
                                        expecting: toy,
                                        maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }

    @Test func wrongMagicThrowsNotAGTurboDirectory() throws {
        let (dir, toy) = try Self.writeToyManifest(["magic": "NOT_GTURBO"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.notAGTurboDirectory) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        }
    }

    @Test func versionTwoThrowsUnsupportedVersion() throws {
        let (dir, toy) = try Self.writeToyManifest(["versionMajor": 2])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unsupportedVersion(let maj, _) = error { return maj == 2 }
            return false
        }
    }

    @Test func unknownFlagThrowsUnknownFlag() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "newFangledOption": true])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unknownFlag(let n) = error { return n == "newFangledOption" }
            return false
        }
    }

    @Test func removedTurboQuantFlagIsRejected() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "turboQuantKV": true,
                                                           "aneSharedExpert": false])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("removed TurboQuant KV")
        }
    }

    @Test func productionManifestRequiresQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.quant is required")
        }
    }

    @Test func productionManifestAcceptsInt4SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.gemma4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 4)
    }

    @Test func productionManifestAcceptsHistoricalInt8SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.gemma4, slot: "sharedExpert", value: 8)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 8)
    }

    @Test func productionManifestRejectsUnsupportedQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.gemma4, slot: "sharedExpert", value: 3)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization")
        }
    }

    @Test func archMismatchThrowsArchMismatch() throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: ["hiddenSize": 4096])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "hiddenSize"
        }
    }

    @Test func nonPageAlignedExpertStrideThrows() throws {
        let (dir, toy) = try Self.writeToyManifest(["expertStride": 1024])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.expertStrideNotPageAligned = error { return true }
            return false
        }
    }

    @Test func missingLayerFileThrowsMissingFile() throws {
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            // intentionally do not list layer_0.bin or layer_1.bin
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.missingFile = error { return true }
            return false
        }
    }

    @Test func acceptsZeroPaddedLayerFilenames() throws {
        // Writer emits packed_experts/layer_%02d.bin; loader should accept either form.
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_00.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_01.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.numLayers == toy.numLayers)
    }

    // MARK: - Shipped-manifest compatibility

    /// Every manifest written before `family` existed must keep decoding exactly
    /// as it did, because the manifest bytes are hashed into
    /// `VerifiedInstallReceipt` and cannot be rewritten in place on a user's disk.
    ///
    /// The JSON below is written out as a literal with the production Gemma
    /// values and the writer's key names (`GTurboJSON.archDict`), rather than
    /// generated from `ArchConfig`, so it cannot agree with a broken encoder —
    /// and it carries **no `family` key**, which is the whole point. Every field
    /// is asserted, because a hand-written `init(from:)` fails per-field: one
    /// mistyped key name silently stops reading one number.
    @Test func shippedGemmaArchDecodesFieldForFieldWithNoFamilyKey() throws {
        let json = """
        {
          "attentionKEqV": true,
          "ffnIntermediate": 2112,
          "finalLogitSoftcap": 30,
          "fullAttentionLayerMask": [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,
                                     0,0,0,0,0,1,0,0,0,0,0,1],
          "fullHeadDim": 512,
          "fullRopeTheta": 1000000,
          "headDim": 256,
          "hiddenActivation": "gelu_pytorch_tanh",
          "hiddenSize": 2816,
          "moeIntermediateSize": 704,
          "numExperts": 128,
          "numFullKVHeads": 2,
          "numHeads": 16,
          "numKVHeads": 8,
          "numLayers": 30,
          "partialRotaryFactor": 0.25,
          "ropeTheta": 10000,
          "slidingWindow": 1024,
          "tieWordEmbeddings": true,
          "topKExperts": 8,
          "vocabSize": 262144
        }
        """
        let a = try JSONDecoder().decode(ManifestArch.self, from: Data(json.utf8))

        #expect(a.hiddenSize == 2816)
        #expect(a.ffnIntermediate == 2112)
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
        // Element by element, not by count-and-popcount. Those two agreed with
        // any permutation of five ones among thirty layers, which is precisely
        // the field where position IS the meaning: element `i` decides whether
        // layer `i` dispatches full or sliding attention. The literal repeats
        // the JSON above deliberately — the claim is that the decoder moved the
        // array across unchanged, so both sides are written out.
        #expect(a.fullAttentionLayerMask == [
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
        ])

        // The absent key defaults to gemma4 — this is what keeps every shipped
        // install loadable.
        #expect(a.variant == .gemma4)

        // And it still validates against the production baseline it describes.
        #expect(ManifestReader.matchArch(a, against: ArchConfig.supported) != nil)
    }

    /// A truncated manifest must fail, not acquire a plausible default. The
    /// hand-written `init(from:)` uses `decode`, never `decodeIfPresent`, for
    /// all 21 pre-existing keys; `family` is the only optional one.
    ///
    /// **One case per key, because the claim is per key.** A hand-written
    /// `init(from:)` fails one field at a time: swapping a single `decode` for
    /// a `decodeIfPresent(... ) ?? 0` stops one number being read and leaves
    /// the other twenty alone. A single-key test asserted the property for
    /// `numLayers` and, by its wording, implied it for twenty keys it never
    /// touched. `Self.mandatoryArchKeys` is the same list `validateArch`'s
    /// coverage test uses, so a key can only be added to one of them.
    @Test(arguments: ManifestReaderTests.mandatoryArchKeys)
    func everyMandatoryArchKeyIsRequired(_ key: String) throws {
        let (dir, toy) = try Self.writeToyManifest(archRemovals: [key])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.json") && detail.contains(key)
        }
    }

    /// And the same for the ten payload keys a bailing manifest carries, which
    /// are decoded from the same container by `BailingMoeV2Extras`.
    @Test(arguments: ManifestReaderTests.mandatoryBailingPayloadKeys)
    func everyMandatoryBailingPayloadKeyIsRequired(_ key: String) throws {
        let (dir, toy) = try Self.writeToyManifest(archRemovals: [key],
                                                   config: .bailingToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.json") && detail.contains(key)
        }
    }

    // MARK: - BailingMoeV2

    /// A bailing manifest clears every check about the FILE against its own
    /// baseline — and is then refused because this build has no kernels for its
    /// family.
    ///
    /// Both halves are the point. `matchArch` succeeding is what proves the
    /// bailing arm of `validateArch` reads the payload and compares it rather
    /// than passing everything after the family gate; the refusal is what
    /// proves a correct, complete bailing install still cannot load here. If
    /// this ever loads cleanly, BailingMoeV2 became executable — that is a
    /// finding, not a test to update.
    @Test func bailingManifestMatchesItsOwnBaselineAndIsStillRefused() throws {
        let (dir, toy) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.bailingMoeV2)],
            config: .bailingToy())
        defer { try? FileManager.default.removeItem(at: dir) }

        // Decoded directly, because `load` refuses these bytes: the assertions
        // below are about what the manifest SAYS, which no gate changes.
        let data = try Data(contentsOf: dir.appendingPathComponent("manifest.json"))
        let decoded = try JSONDecoder().decode(Manifest.self, from: data)
        #expect(decoded.arch.variant == toy.variant)
        // The shared-expert width, not the dense one — both are present in the
        // manifest and only one is `ffnIntermediate`.
        #expect(decoded.arch.ffnIntermediate == 32)
        guard case .bailingMoeV2(let extras) = decoded.arch.variant else {
            Issue.record("decoded variant is not bailingMoeV2")
            return
        }
        #expect(extras.denseIntermediateSize == 320)

        // Every field-by-field comparison against its own baseline passes...
        #expect(ManifestReader.matchArch(decoded.arch, against: [toy]) != nil)

        // ...and the load is refused anyway, for the one reason left.
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.architectureNotExecutable(let family) = error else {
                return false
            }
            return family == "bailingMoeV2"
        }
    }

    /// **The gate is keyed on the family, not on a list of known dimensions.**
    ///
    /// This manifest is a BailingMoeV2 install of a checkpoint nobody has
    /// transcribed a baseline for — a different Ling size, a fine-tune, the
    /// next release. It matches no entry in `ArchConfig.real`, so a gate keyed
    /// on `real` never fires for it; it matches no entry in `supported`
    /// either, so `validateArch` runs against `expecting:` — and here the
    /// caller passes that manifest's own baseline, which is the case a caller
    /// can talk itself into. With every other check satisfied, the only thing
    /// that can refuse it is the architectural fact that this binary contains
    /// no BailingMoeV2 kernels.
    ///
    /// Keyed on `real`, this test loads the model and then traps at the first
    /// attention dispatch.
    @Test func anUnknownBailingCheckpointIsRefusedThoughItMatchesNothingInReal() throws {
        // `bailingToy` with two more layers and a wider hidden size: still
        // BailingMoeV2, still internally consistent, and equal to nothing in
        // `real`.
        let unknown = ArchConfig.bailingToy(numLayers: 4, hiddenSize: 96)
        #expect(!ArchConfig.real.contains(unknown))
        #expect(!ArchConfig.supported.contains(unknown))

        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.bailingMoeV2)],
            config: unknown)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Precondition of the test: the manifest really does describe an
        // architecture the `real` list has never heard of, so a gate keyed
        // there has nothing to fire on.
        let data = try Data(contentsOf: dir.appendingPathComponent("manifest.json"))
        let decoded = try JSONDecoder().decode(Manifest.self, from: data)
        #expect(ManifestReader.matchArch(decoded.arch, against: ArchConfig.real) == nil)

        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.architectureNotExecutable(let family) = error else {
                return false
            }
            return family == "bailingMoeV2"
        }
    }

    /// A Ling manifest checked against the Gemma baseline must report the
    /// family, not whichever core field happens to differ first. "hiddenSize
    /// 64, expected 2816" sends the reader hunting for a corrupt Gemma install.
    @Test func familyMismatchIsReportedAsFamily() throws {
        let (dir, _) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.bailingMoeV2)],
            config: .bailingToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .gemma4_26B_A4B)
        } throws: { error in
            guard case let ModelError.archMismatch(field, expected, actual) = error else {
                return false
            }
            return field == "family" && expected == "gemma4" && actual == "bailingMoeV2"
        }
    }

    /// The reverse direction: a Gemma manifest offered to a bailing baseline is
    /// also a family mismatch, and gets there before any core comparison.
    @Test func gemmaManifestAgainstBailingBaselineIsAFamilyMismatch() throws {
        let (dir, _) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .bailingToy())
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "family"
        }
    }

    /// The payload keys are flat inside `arch` and decoded as part of the
    /// variant, so an omitted one is `keyNotFound` at decode time. Had they been
    /// modelled as optional properties on `ManifestArch`, this manifest would
    /// decode to `nil` and then compare equal to the baseline's `nil`.
    @Test func bailingManifestMissingPayloadKeyThrows() throws {
        let (dir, toy) = try Self.writeToyManifest(
            archRemovals: ["denseIntermediateSize"],
            config: .bailingToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("denseIntermediateSize")
        }
    }

    /// A payload field that differs is reported by its own name, so the bailing
    /// arm is doing real work rather than passing everything after the family
    /// gate.
    @Test func bailingPayloadMismatchNamesTheField() throws {
        let (dir, toy) = try Self.writeToyManifest(
            archOverrides: ["topkGroup": 3],
            config: .bailingToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, expected, actual) = error else {
                return false
            }
            return field == "topkGroup" && expected == "2" && actual == "3"
        }
    }

    // MARK: - Per-architecture quantization

    /// **The assertion that proves the router gate was made per-architecture
    /// rather than merely widened.**
    ///
    /// Gemma's router kernels take `device const uint8_t*` and index one byte
    /// per weight; a genuinely 4-bit router blob is read over twice its length,
    /// out of bounds, producing garbage logits with no assertion anywhere. If
    /// this test goes green after a change to the bit-width table, the table was
    /// widened to `[4, 8]` and Gemma can now install with a 4-bit router.
    ///
    /// It is separate from `productionManifestRejectsUnsupportedQuantMetadata`
    /// on purpose: that one perturbs the shared expert, so it would stay green
    /// through exactly this regression.
    @Test func gemmaRejectsAFourBitRouter() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.gemma4, slot: "router", value: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization for router")
        }
    }

    /// The mirror image: Ling's router really is 4-bit, so an 8-bit claim over
    /// those bytes must be refused too. Without this, "per-architecture" could
    /// mean "the union of both", which admits every wrong combination.
    @Test func bailingRejectsAnEightBitRouter() throws {
        let (dir, toy) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.bailingMoeV2, slot: "router", value: 8)],
            config: .bailingToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization for router")
        }
    }

    /// Gemma's 8-bit shared expert is a real allowance for historical installs.
    /// Ling has no such history, so it does not inherit the allowance.
    @Test func bailingRejectsAnEightBitSharedExpert() throws {
        let (dir, toy) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.bailingMoeV2, slot: "sharedExpert", value: 8)],
            config: .bailingToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization for sharedExpert")
        }
    }

    /// **The bit-width table must not be skippable.** `manifest.quant` is an
    /// optional property, so without this gate a real Ling install omits the
    /// block and never reaches `validateQuant` at all — the per-architecture
    /// table, which is the whole point of the change, is bypassed by leaving
    /// something out.
    ///
    /// Keyed off `ArchConfig.real` rather than `supported` precisely because
    /// Ling is deliberately out of `supported` until its kernels land: keying
    /// off executability would make "no kernels yet" also mean "no quantization
    /// metadata needed", which is not a thing anyone decided.
    @Test func realLingInstallWithoutQuantIsRejected() throws {
        let (dir, config) = try Self.writeToyManifest(config: .lingMini2_0_4bit)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.quant is required")
                && detail.contains("bailingMoeV2")
        }
    }

    /// The same install with the quant block Ling really has gets *past* the
    /// mandatory-block gate — it is refused for want of kernels instead, which
    /// is a different verdict from a different check. That is what makes the
    /// test above a gate on the missing block rather than on Ling itself: if
    /// the two were the same rule, this would still say "quant is required".
    ///
    /// Deliberately asserts the error rather than a successful load. A
    /// complete, correct Ling install loading in a build with no BailingMoeV2
    /// kernels is the thing this milestone must never allow — see
    /// `ArchConfigLingTests.aRealLingInstallCannotLoadWithoutItsKernels`.
    @Test func realLingInstallWithItsOwnQuantClearsTheQuantGate() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.bailingMoeV2)],
            config: .lingMini2_0_4bit)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.architectureNotExecutable(let family) = error else {
                return false
            }
            return family == "bailingMoeV2"
        }
    }

    /// A real Ling install cannot borrow Gemma's 8-bit router either — the
    /// mandatory block is validated against Ling's own row of the table.
    @Test func realLingInstallWithGemmasRouterWidthIsRejected() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.bailingMoeV2, slot: "router", value: 8)],
            config: .lingMini2_0_4bit)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization for router")
        }
    }

    /// A toy baseline is not a real checkpoint, so it is not held to the
    /// mandatory-block rule. Stated as a decision rather than left implicit,
    /// because it is what lets most of this suite write manifests without one.
    ///
    /// Asserted through a bailing toy — which the family gate refuses — so what
    /// this pins is that the refusal is "no kernels" and NOT "quant is
    /// required". A toy of an executable family reaching a clean load is
    /// covered by `loadsValidManifest`, which writes no quant block either.
    @Test func quantStaysOptionalForAToyArchitecture() throws {
        let (dir, toy) = try Self.writeToyManifest(config: .bailingToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            // Specifically not `indexCorrupt("manifest.quant is required...")`:
            // the mandatory-block rule is keyed on `real`, and a toy is not one.
            guard case ModelError.architectureNotExecutable = error else { return false }
            return true
        }

        // And the gemma toy every other test in this suite uses really does load
        // with no quant block, which is the other half of the same claim.
        let (gemmaDir, gemmaToy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: gemmaDir) }
        #expect(try ManifestReader.load(directoryURL: gemmaDir,
                                        expecting: gemmaToy).quant == nil)
    }

    // MARK: - Every guard in validateQuant is load-bearing

    /// One case per (slot, guard) pair `validateQuant` enforces for Gemma.
    /// Deleting any row of the `slots` table, or any one of the four
    /// non-bit-width conditions, must turn a case here red.
    ///
    /// This is the half that was missing. Three of the five slots —
    /// `embedding`, `attention`, `routedExpert` — are 4-bit in **both**
    /// families, so every earlier fixture wrote 4 into them and every earlier
    /// test would have stayed green with those rows deleted outright. A slot
    /// nobody ever writes a wrong width into is not being checked; it is being
    /// agreed with.
    @Test(arguments: ManifestReaderTests.gemmaQuantPerturbations)
    func perturbingAnyGemmaQuantGuardIsReported(_ p: QuantSlotPerturbation) throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.gemma4, slot: p.slot,
                                      field: p.field, value: p.value.json)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail == "unsupported quantization for \(p.slot)"
        }
    }

    /// The same for BailingMoeV2, against a real Ling install so the block is
    /// checked on the mandatory path rather than the optional one.
    ///
    /// Every slot has an expectation for both families, so every slot gets a
    /// case for both: "per-architecture" has to mean each family's own row is
    /// consulted, not that one family's row is consulted twice.
    @Test(arguments: ManifestReaderTests.bailingQuantPerturbations)
    func perturbingAnyBailingQuantGuardIsReported(_ p: QuantSlotPerturbation) throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.bailingMoeV2, slot: p.slot,
                                      field: p.field, value: p.value.json)],
            config: .lingMini2_0_4bit)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail == "unsupported quantization for \(p.slot)"
        }
    }

    /// The tables above are worth only what they cover, so their coverage is
    /// pinned: every slot, and every one of the five conditions, for both
    /// families. A guard added to `validateQuant` without a row here would
    /// arrive untested, which is how four of the five arrived.
    @Test func theQuantTablesCoverEverySlotAndEveryGuard() {
        let slots = ["attention", "embedding", "routedExpert", "router", "sharedExpert"]
        let guards = ["biasType", "groupSize", "scaleType", "scheme", "weightBits"]

        for table in [Self.gemmaQuantPerturbations, Self.bailingQuantPerturbations] {
            #expect(Set(table.map(\.slot)).sorted() == slots)
            #expect(Set(table.map(\.field)).sorted() == guards)
        }
    }

    // MARK: - `expecting:` is a fallback baseline

    /// A manifest matching a `supported` architecture loads whatever baseline
    /// the caller passed — `validateArch` is skipped entirely.
    ///
    /// Pinned because it is surprising and because it is deliberate:
    /// `supported` is the set this build ships kernels for, and executability
    /// is the property that decides. `expecting:` exists so a manifest that
    /// matches *nothing* can be refused with a field name instead of a shrug.
    /// If this ever needs to be a real demand, it is a decision to make and a
    /// test to rewrite, not a bug to notice in production.
    @Test func manifestMatchingSupportedLoadsAgainstAnyBaseline() throws {
        let (dir, _) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.gemma4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }

        // A baseline from a different family, different in all 21 core fields.
        let m = try ManifestReader.load(directoryURL: dir, expecting: .bailingToy())
        #expect(m.arch.variant == .gemma4)
    }

    /// ...but the *quant* block is still checked against the architecture the
    /// manifest matched, not against that baseline.
    ///
    /// This is the `matchedReal` half of `matchedReal ?? expected`. Replace it
    /// with plain `expected` and this manifest is validated against
    /// `bailingToy`'s row, which accepts a 4-bit router — so a production Gemma
    /// install with a 4-bit router blob loads and reads its router weights out
    /// of bounds.
    @Test func quantIsCheckedAgainstTheMatchedArchitectureNotTheCallersBaseline() throws {
        let (dir, _) = try Self.writeToyManifest(
            ["quant": Self.quantBlock(.gemma4, slot: "router", value: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }

        // `bailingToy` accepts a 4-bit router; `gemma4_26B_A4B`, which these
        // bytes actually describe, does not.
        #expect(ArchConfig.bailingToy().variant.acceptedQuantBits.router == [4])
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .bailingToy())
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail == "unsupported quantization for router"
        }
    }

    // MARK: - Untrusted numbers must throw, not trap

    /// A manifest is a file this process did not write, so an out-of-range
    /// number in it has to be a rejection and not a process abort.
    ///
    /// `fullAttentionLayerMask` was narrowed with
    /// `a.fullAttentionLayerMask.map { UInt8($0) }`, and `UInt8.init(_:)`
    /// **traps**: either value below aborted the whole process — the app, mid
    /// load — instead of reporting a corrupt install. Both are asserted
    /// because they trap for different reasons, and a fix that clamps rather
    /// than widens would still lose one of them.
    @Test(arguments: [-1, 999, Int(Int32.min)])
    func anOutOfByteRangeLayerMaskIsRejectedRatherThanTrapping(_ value: Int) throws {
        let (dir, toy) = try Self.writeToyManifest(
            archOverrides: ["fullAttentionLayerMask": [0, value]])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, actual) = error else {
                return false
            }
            // Reported in `Int`, so the offending value is legible in the
            // message rather than wrapped into something plausible.
            return field == "fullAttentionLayerMask" && actual.contains("\(value)")
        }
    }

    /// The top-level `numLayers` is a different number from `arch.numLayers`
    /// and nothing compares it, so it reaches `for L in 0..<m.numLayers`
    /// straight from the file. A negative bound is a **trap** in Swift, not an
    /// empty loop.
    @Test func aNegativeTopLevelLayerCountIsRejectedRatherThanTrapping() throws {
        let (dir, toy) = try Self.writeToyManifest(["numLayers": -1])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("numLayers")
        }
    }

    // MARK: - The `family` key: absent, null, unknown

    /// An absent `family` is every manifest written before the key existed and
    /// must keep decoding as Gemma. The manifest is re-read from disk first so
    /// the assertion cannot pass because the helper quietly started writing one.
    @Test func absentFamilyKeyDecodesAsGemma() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let raw = try JSONSerialization.jsonObject(
            with: Data(contentsOf: dir.appendingPathComponent("manifest.json")))
        let archDict = (raw as? [String: Any])?["arch"] as? [String: Any]
        #expect(archDict?["family"] == nil, "the fixture must not write a family key")

        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.arch.variant == .gemma4)
    }

    /// `"family": null` is not the same as no family, and `decodeIfPresent`
    /// cannot tell them apart. Absent means "predates the key"; null means a
    /// writer emitted a family it could not name, which is malformed.
    @Test func explicitNullFamilyIsRejected() throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: ["family": NSNull()])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("family")
        }
    }

    /// And a family this build has never heard of is refused rather than
    /// adopted by whichever arm tolerates the remaining keys.
    @Test func unknownFamilyIsRejected() throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: ["family": "qwen3Moe"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt = error else { return false }
            return true
        }
    }

    // MARK: - Every comparison in validateArch is load-bearing

    /// One case per field `validateArch` compares for Gemma. Deleting any one
    /// `check` line must turn a case here red.
    ///
    /// Table-driven and parameterised rather than a loop inside one test, so a
    /// deleted comparison names the field it stopped checking instead of
    /// reddening one opaque test that has to be read to find out which.
    @Test(arguments: ManifestReaderTests.gemmaArchPerturbations)
    func perturbingAnyGemmaArchFieldIsReported(_ p: ArchFieldPerturbation) throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: [p.field: p.value.json])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == p.field
        }
    }

    /// The same for BailingMoeV2, across all 21 core fields **and** all ten
    /// payload fields.
    ///
    /// The core half is the one that was missing: `numFullKVHeads`,
    /// `fullHeadDim`, `slidingWindow`, `finalLogitSoftcap`, `ropeTheta` and
    /// `attentionKEqV` were once compared only on the Gemma arm, so a corrupted
    /// Ling manifest carrying the wrong head geometry — the geometry a
    /// 100%-full-attention model actually dispatches with — validated clean and
    /// trapped later.
    @Test(arguments: ManifestReaderTests.bailingArchPerturbations)
    func perturbingAnyBailingArchFieldIsReported(_ p: ArchFieldPerturbation) throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: [p.field: p.value.json],
                                                   config: .bailingToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == p.field
        }
    }

    /// The tables above are only worth what they cover, so their coverage is
    /// pinned too: a comparison added to `validateArch` without a row here would
    /// otherwise arrive untested, which is exactly how the ten payload
    /// comparisons and the six relocated core ones arrived.
    ///
    /// `family` is absent from both lists on purpose — it cannot be perturbed
    /// in place without becoming a different family, and it has its own tests.
    @Test func theTablesCoverEveryFieldValidateArchCompares() {
        let core = Self.mandatoryArchKeys
        let payload = Self.mandatoryBailingPayloadKeys

        #expect(core.count == 21)
        #expect(payload.count == 10)
        #expect(Self.gemmaArchPerturbations.map(\.field).sorted() == core)
        #expect(Self.bailingArchPerturbations.map(\.field).sorted()
                == (core + payload).sorted())
    }

    /// The 21 `arch` keys every manifest must carry — the same list the
    /// perturbation tables are checked against and the same list
    /// `everyMandatoryArchKeyIsRequired` removes one at a time, so "compared by
    /// `validateArch`" and "required by the decoder" cannot drift apart.
    ///
    /// Sorted, because `theTablesCoverEveryFieldValidateArchCompares` compares
    /// it against a sorted table.
    static let mandatoryArchKeys = [
        "attentionKEqV", "ffnIntermediate", "finalLogitSoftcap",
        "fullAttentionLayerMask", "fullHeadDim", "fullRopeTheta", "headDim",
        "hiddenActivation", "hiddenSize", "moeIntermediateSize", "numExperts",
        "numFullKVHeads", "numHeads", "numKVHeads", "numLayers",
        "partialRotaryFactor", "ropeTheta", "slidingWindow",
        "tieWordEmbeddings", "topKExperts", "vocabSize",
    ]

    /// The ten flat payload keys a `bailingMoeV2` manifest adds.
    static let mandatoryBailingPayloadKeys = [
        "denseIntermediateSize", "firstKDenseReplace", "nGroup",
        "normTopkProb", "numSharedExperts", "routedScalingFactor",
        "routerEnableExpertBias", "scoreFunction", "topkGroup", "useQKNorm",
    ]

    // MARK: - Perturbation tables

    /// Every value differs from `ArchConfig.gemma4Toy()`'s, and none of them
    /// turns the toy into the production Gemma baseline — a manifest that
    /// matched `supported` would skip `validateArch` entirely and the case would
    /// pass for the wrong reason.
    static let gemmaArchPerturbations: [ArchFieldPerturbation] = [
        .init("hiddenSize", .int(128)),
        .init("ffnIntermediate", .int(512)),
        .init("moeIntermediateSize", .int(256)),
        .init("numHeads", .int(8)),
        .init("numKVHeads", .int(4)),
        .init("numFullKVHeads", .int(2)),
        .init("headDim", .int(32)),
        .init("fullHeadDim", .int(64)),
        .init("vocabSize", .int(2048)),
        .init("slidingWindow", .int(512)),
        .init("finalLogitSoftcap", .double(50.0)),
        .init("ropeTheta", .double(20_000.0)),
        .init("fullRopeTheta", .double(500_000.0)),
        .init("partialRotaryFactor", .double(0.5)),
        .init("numLayers", .int(3)),
        .init("numExperts", .int(16)),
        .init("topKExperts", .int(4)),
        .init("tieWordEmbeddings", .bool(false)),
        .init("attentionKEqV", .bool(false)),
        .init("hiddenActivation", .string("silu")),
        .init("fullAttentionLayerMask", .intArray([1, 1])),
    ]

    /// Every value differs from `ArchConfig.bailingToy()`'s. Note
    /// `slidingWindow` and `finalLogitSoftcap`: Ling's 0 is a definite "none",
    /// not a placeholder, so a manifest claiming 256 or 30 is corrupt and must
    /// say so.
    static let bailingArchPerturbations: [ArchFieldPerturbation] = [
        .init("hiddenSize", .int(128)),
        .init("ffnIntermediate", .int(64)),
        .init("moeIntermediateSize", .int(64)),
        .init("numHeads", .int(8)),
        .init("numKVHeads", .int(4)),
        .init("numFullKVHeads", .int(4)),
        .init("headDim", .int(32)),
        .init("fullHeadDim", .int(32)),
        .init("vocabSize", .int(2048)),
        .init("slidingWindow", .int(256)),
        .init("finalLogitSoftcap", .double(30.0)),
        .init("ropeTheta", .double(10_000.0)),
        .init("fullRopeTheta", .double(10_000.0)),
        .init("partialRotaryFactor", .double(0.25)),
        .init("numLayers", .int(3)),
        .init("numExperts", .int(16)),
        .init("topKExperts", .int(4)),
        .init("tieWordEmbeddings", .bool(true)),
        .init("attentionKEqV", .bool(true)),
        .init("hiddenActivation", .string("gelu_pytorch_tanh")),
        .init("fullAttentionLayerMask", .intArray([0, 1])),

        .init("denseIntermediateSize", .int(640)),
        .init("firstKDenseReplace", .int(2)),
        .init("numSharedExperts", .int(2)),
        .init("nGroup", .int(8)),
        .init("topkGroup", .int(3)),
        .init("routedScalingFactor", .double(1.5)),
        .init("normTopkProb", .bool(false)),
        .init("scoreFunction", .string("softmax")),
        .init("routerEnableExpertBias", .bool(false)),
        .init("useQKNorm", .bool(false)),
    ]

    // MARK: - Quant perturbation tables

    /// One quant guard per row, spread across the slots so that a deleted slot
    /// row and a deleted condition are both caught, and caught separately.
    ///
    /// The widths are the ones Gemma's kernels cannot read: `embedding`,
    /// `attention` and `routedExpert` are 4-bit only, the router is 8-bit only,
    /// and the shared expert allows 4 or 8 — so 3, a width nothing supports, is
    /// what perturbs it.
    static let gemmaQuantPerturbations: [QuantSlotPerturbation] = [
        .init("embedding", "weightBits", .int(8)),
        .init("attention", "weightBits", .int(8)),
        .init("router", "weightBits", .int(4)),
        .init("sharedExpert", "weightBits", .int(3)),
        .init("routedExpert", "weightBits", .int(8)),

        .init("embedding", "scheme", .string("symmetric")),
        .init("attention", "scaleType", .string("fp16")),
        .init("router", "biasType", .string("fp16")),
        .init("routedExpert", "groupSize", .int(Quantization.groupSize * 2)),
    ]

    /// Ling is 4-bit in every slot, router included and shared expert included
    /// — it has no historical 8-bit installs to accommodate — so 8 is the wrong
    /// width everywhere. The four non-width guards are package-wide and are
    /// perturbed on the same slots as above.
    static let bailingQuantPerturbations: [QuantSlotPerturbation] = [
        .init("embedding", "weightBits", .int(8)),
        .init("attention", "weightBits", .int(8)),
        .init("router", "weightBits", .int(8)),
        .init("sharedExpert", "weightBits", .int(8)),
        .init("routedExpert", "weightBits", .int(8)),

        .init("embedding", "scheme", .string("symmetric")),
        .init("attention", "scaleType", .string("fp16")),
        .init("router", "biasType", .string("fp16")),
        .init("routedExpert", "groupSize", .int(Quantization.groupSize * 2)),
    ]
}

/// One field of one `manifest.quant` slot set to a value no architecture
/// accepts.
struct QuantSlotPerturbation: Sendable, CustomStringConvertible {
    let slot: String
    let field: String
    let value: ArchFieldValue

    init(_ slot: String, _ field: String, _ value: ArchFieldValue) {
        self.slot = slot
        self.field = field
        self.value = value
    }

    var description: String { "\(slot).\(field) = \(value)" }
}

/// One `arch` key set to a value the baseline does not have.
///
/// `Sendable`, and therefore usable as a `@Test(arguments:)` case list, which a
/// `[String: Any]` table would not be.
struct ArchFieldPerturbation: Sendable, CustomStringConvertible {
    let field: String
    let value: ArchFieldValue

    init(_ field: String, _ value: ArchFieldValue) {
        self.field = field
        self.value = value
    }

    var description: String { "\(field) = \(value)" }
}

/// The JSON scalar shapes the manifest's `arch` object uses.
enum ArchFieldValue: Sendable, CustomStringConvertible {
    case int(Int)
    case double(Double)
    case bool(Bool)
    case string(String)
    case intArray([Int])

    var json: Any {
        switch self {
        case .int(let v): return v
        case .double(let v): return v
        case .bool(let v): return v
        case .string(let v): return v
        case .intArray(let v): return v
        }
    }

    var description: String { String(describing: json) }
}

extension ArchConfig {
    /// Tiny baseline used across the loader tests. 2 layers (both full), hidden 64,
    /// vocab 1024, 8 experts. Numbers are intentionally toy.
    static func gemma4Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 256,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 256,
            finalLogitSoftcap: 30.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 1_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "gelu_pytorch_tanh",
            variant: .gemma4
        )
    }

    /// Tiny BailingMoeV2 baseline. Shaped like Ling — all-full attention, a
    /// dense leading layer, sigmoid grouped routing — but with toy sizes, and
    /// deliberately NOT in `ArchConfig.supported`, so it exercises the bailing
    /// arm of validation without claiming this build can execute it.
    ///
    /// `intermediateSize` (shared expert) and `denseIntermediateSize` are given
    /// *different* toy values, unlike the real config where both moe widths are
    /// 512, so a test that confuses the two fails.
    ///
    /// `numLayers` and `hiddenSize` are parameters so a test can build a
    /// *second*, differently-shaped BailingMoeV2 architecture — the "unknown
    /// checkpoint" case, which must be refused for its family rather than for
    /// its dimensions. Both defaults are the original toy values, so every
    /// existing caller is unchanged.
    static func bailingToy(numLayers: Int = 2, hiddenSize: Int = 64) -> ArchConfig {
        ArchConfig(
            hiddenSize: hiddenSize,
            intermediateSize: 32,
            moeIntermediateSize: 32,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 16,
            fullHeadDim: 16,
            vocabSize: 1024,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 600_000.0,
            fullRopeTheta: 600_000.0,
            partialRotaryFactor: 0.5,
            numLayers: numLayers,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [UInt8](repeating: 1, count: numLayers),
            hiddenActivation: "silu",
            variant: .bailingMoeV2(BailingMoeV2Extras(
                denseIntermediateSize: 320,
                firstKDenseReplace: 1,
                numSharedExperts: 1,
                nGroup: 4,
                topkGroup: 2,
                routedScalingFactor: 2.5,
                normTopkProb: true,
                scoreFunction: "sigmoid",
                routerEnableExpertBias: true,
                useQKNorm: true))
        )
    }
}
