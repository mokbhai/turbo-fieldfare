import Foundation
import Testing
@testable import TurboFieldfareRepackCore

/// `ArchInfo.load` is the first thing that touches a downloaded `config.json`,
/// and until this suite existed it had no direct coverage at all: every
/// assertion about it was indirect, through the repack plan it feeds.
///
/// That matters most for the failure mode this suite is really about. The
/// parse is the only place a wrong architecture number can be *invented*.
/// Nothing downstream catches one, because `validateArch` compares the manifest
/// against a hand-written `ArchConfig` — if the parser and the constant are
/// wrong the same way, the model installs, validates and runs, and the only
/// symptom is bad output.
@Suite
struct ArchInfoTests {

    // MARK: - Gemma, unmoved

    /// The dispatcher must land Gemma on exactly the parse it had before. The
    /// frozen-layout goldens all descend from this one call, so they redden
    /// together and blame the layout; this test names the actual cause.
    @Test func gemmaConfigParsesThroughTheGemmaBranch() throws {
        let directory = temporaryRoot("gemma-arch")
        defer { try? FileManager.default.removeItem(atPath: directory) }
        _ = try SyntheticSnapshot.build(at: directory)

        let arch = try ArchInfo.load(
            configPath: (directory as NSString).appendingPathComponent("config.json"))

        let expected = SyntheticSnapshot.Arch()
        #expect(arch.hiddenSize == expected.hidden)
        #expect(arch.intermediateSize == expected.intermediate)
        #expect(arch.moeIntermediateSize == expected.moeIntermediate)
        #expect(arch.numHeads == expected.numHeads)
        #expect(arch.numKVHeads == expected.numKVHeads)
        #expect(arch.numFullKVHeads == expected.numGlobalKVHeads)
        #expect(arch.headDim == expected.headDim)
        #expect(arch.fullHeadDim == expected.globalHeadDim)
        #expect(arch.vocabSize == expected.vocab)
        #expect(arch.slidingWindow == expected.slidingWindow)
        #expect(arch.finalLogitSoftcap == 30.0)
        #expect(arch.ropeTheta == 10_000.0)
        #expect(arch.fullRopeTheta == 1_000_000.0)
        #expect(arch.partialRotaryFactor == 0.25)
        #expect(arch.numLayers == expected.numLayers)
        #expect(arch.numExperts == expected.numExperts)
        #expect(arch.topKExperts == expected.topK)
        #expect(arch.tieWordEmbeddings == true)
        #expect(arch.attentionKEqV == true)
        // layer 0 sliding, layer 1 full — derived from `layer_types`.
        #expect(arch.fullAttentionLayerMask == [0, 1])
        #expect(arch.hiddenActivation == "gelu_pytorch_tanh")
        #expect(arch.variant == .gemma4)
    }

    /// Gemma's config carries both `model_type` and `architectures`; the
    /// architectures-only fallback exists so this parse and the pre-download
    /// gate agree, and it must not accidentally reroute Gemma.
    @Test func gemmaDispatchesOnArchitecturesWhenModelTypeIsAbsent() throws {
        let directory = temporaryRoot("gemma-arch-only")
        defer { try? FileManager.default.removeItem(atPath: directory) }
        _ = try SyntheticSnapshot.build(at: directory)
        let path = (directory as NSString).appendingPathComponent("config.json")
        var config = try jsonObject(at: path)
        config.removeValue(forKey: "model_type")
        try write(config, to: path)

        let arch = try ArchInfo.load(configPath: path)
        #expect(arch.variant == .gemma4)
    }

    /// **The production Gemma config, parsed field for field.**
    ///
    /// `gemmaConfigParsesThroughTheGemmaBranch` above uses the toy synthetic
    /// snapshot, whose numbers are all small and all positive — so it would stay
    /// green through a bound that happened to reject a real value. These are the
    /// published `google/gemma-4-26b-a4b` numbers, transcribed, and they are
    /// what the runtime's `ArchConfig.gemma4_26B_A4B` compares every installed
    /// manifest against: if the parse of a valid Gemma config moves by one
    /// field, every repacked install stops loading.
    ///
    /// Written when `loadGemma4` was moved onto the strict accessors the
    /// BailingMoeV2 branch already used. Nothing about a *valid* config may
    /// change; that is the claim, and this is where it is checked.
    @Test func productionGemmaConfigParsesUnchanged() throws {
        let directory = temporaryRoot("gemma-production")
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try writeGemmaConfig(to: directory)

        let arch = try ArchInfo.load(configPath: path)

        #expect(arch.hiddenSize == 2816)
        #expect(arch.intermediateSize == 2112)
        #expect(arch.moeIntermediateSize == 704)
        #expect(arch.numHeads == 16)
        #expect(arch.numKVHeads == 8)
        #expect(arch.numFullKVHeads == 2)
        #expect(arch.headDim == 256)
        #expect(arch.fullHeadDim == 512)
        #expect(arch.vocabSize == 262_144)
        #expect(arch.slidingWindow == 1024)
        #expect(arch.finalLogitSoftcap == 30.0)
        #expect(arch.ropeTheta == 10_000.0)
        #expect(arch.fullRopeTheta == 1_000_000.0)
        #expect(arch.partialRotaryFactor == 0.25)
        #expect(arch.numLayers == 30)
        #expect(arch.numExperts == 128)
        #expect(arch.topKExperts == 8)
        #expect(arch.tieWordEmbeddings == true)
        #expect(arch.attentionKEqV == true)
        #expect(arch.hiddenActivation == "gelu_pytorch_tanh")
        #expect(arch.variant == .gemma4)
        #expect(arch.fullAttentionLayerMask == [
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
        ])
    }

    // MARK: - Gemma type strictness
    //
    // One case per coercion the Gemma branch used to perform. Every one of these
    // parsed *successfully* before, with a number nobody wrote — which is the
    // failure mode this whole suite exists for, since nothing downstream catches
    // an invented value.

    /// `(tc[k] as? NSNumber)?.intValue` accepted a JSON boolean, because
    /// `JSONSerialization` returns booleans as `NSNumber` too: `hidden_size:
    /// true` used to parse as a hidden size of 1.
    @Test func gemmaConfigRejectsABooleanWhereANumberIsMeant() throws {
        #expect(throws: RepackError.self) {
            _ = try loadGemma(overriding: ["hidden_size": true], tag: "gemma-bool-int")
        }
    }

    /// `.intValue` truncates, so `num_hidden_layers: 30.5` used to produce a
    /// 30-layer model from a config that says something else.
    @Test func gemmaConfigRejectsANonIntegerLayerCount() throws {
        #expect(throws: RepackError.self) {
            _ = try loadGemma(overriding: ["num_hidden_layers": 30.5],
                              tag: "gemma-fractional-layers")
        }
    }

    /// A number spelled as a string is not a number — this one already threw,
    /// and is here so the table above cannot be mistaken for the whole rule.
    @Test func gemmaConfigRejectsAStringWhereANumberIsMeant() throws {
        #expect(throws: RepackError.self) {
            _ = try loadGemma(overriding: ["hidden_size": "2816"], tag: "gemma-string-int")
        }
    }

    /// The trap: `num_hidden_layers: -1` reached the planner's
    /// `arch.numLayers`-sized loops and allocations, and a negative count TRAPS
    /// rather than throwing — the repacker aborted instead of reporting an
    /// invalid config. Zero is in the table because it does not trap: it yields
    /// a zero-layer model whose masks are empty and whose manifest is nonsense.
    @Test(arguments: [-1, 0, ArchInfo.maxPlausibleDimension + 1])
    func gemmaConfigRejectsAnOutOfRangeLayerCount(_ layers: Int) throws {
        #expect(throws: RepackError.self) {
            _ = try loadGemma(overriding: ["num_hidden_layers": layers],
                              tag: "gemma-layers-\(layers)")
        }
    }

    /// The same for every width and count. None of these traps on its own, but
    /// each propagates into the manifest and then into buffer sizing.
    @Test(arguments: ["hidden_size", "intermediate_size", "moe_intermediate_size",
                      "num_attention_heads", "num_key_value_heads",
                      "num_global_key_value_heads", "head_dim", "global_head_dim",
                      "vocab_size", "num_experts", "top_k_experts"])
    func gemmaConfigRejectsANonPositiveDimension(_ key: String) throws {
        #expect(throws: RepackError.self) {
            _ = try loadGemma(overriding: [key: -1], tag: "gemma-negative-\(key)")
        }
        #expect(throws: RepackError.self) {
            _ = try loadGemma(overriding: [key: 0], tag: "gemma-zero-\(key)")
        }
    }

    /// `sliding_window` is the one count where zero is a real answer — it means
    /// the model has no sliding window, which is what a fully-global variant
    /// would say, and it is exactly how Ling's arch records the same fact.
    /// Guarded one step looser, and that looseness is pinned rather than left to
    /// be discovered.
    @Test func gemmaAcceptsAZeroSlidingWindowButNotANegativeOne() throws {
        let arch = try loadGemma(overriding: ["sliding_window": 0], tag: "gemma-zero-swa")
        #expect(arch.slidingWindow == 0)

        #expect(throws: RepackError.self) {
            _ = try loadGemma(overriding: ["sliding_window": -1], tag: "gemma-negative-swa")
        }
    }

    /// The keys with historical defaults keep them when ABSENT — Gemma configs
    /// in the wild omit some of these — but a value stated in the wrong type is
    /// no longer coerced into one. `tie_word_embeddings: 1` used to satisfy
    /// `as? Bool` and read as `true`; `hidden_activation: true` used to fall
    /// through to `"gelu_pytorch_tanh"` and report nothing.
    ///
    /// A loop rather than `@Test(arguments:)` because a `[String: Any]` case
    /// list is not `Sendable`; each iteration names its own key on failure.
    @Test func gemmaConfigRejectsAWrongTypedOptionalKey() throws {
        let overrides: [(String, Any)] = [
            ("tie_word_embeddings", 1),
            ("attention_k_eq_v", 0),
            ("hidden_activation", true),
            ("layer_types", 5),
            ("rope_parameters", "default"),
        ]
        for (key, value) in overrides {
            #expect(throws: RepackError.self,
                    "\(key) stated as \(value) was coerced instead of refused") {
                _ = try loadGemma(overriding: [key: value],
                                  tag: "gemma-wrongtype-\(key)")
            }
        }
    }

    /// An explicit `null` means the same thing as omitting the key.
    ///
    /// `JSONSerialization` decodes JSON `null` to `NSNull()`, which is not
    /// `nil`, so testing absence with `values[k] == nil` would route
    /// `"key": null` to the strict reader and throw — rejecting configs that
    /// parse on main, for a spelling upstream uses to mean "unset".
    ///
    /// Deliberately the opposite of `manifest.json`'s `family` key, where an
    /// explicit null IS rejected: that file is written and read by this
    /// package, so a null there is malformed rather than conventional.
    @Test func gemmaOptionalKeysTreatAnExplicitNullAsAbsent() throws {
        let arch = try loadGemma(
            overriding: ["tie_word_embeddings": NSNull(),
                         "attention_k_eq_v": NSNull(),
                         "hidden_activation": NSNull(),
                         "rope_parameters": NSNull()],
            tag: "gemma-null-optionals")
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.attentionKEqV == false)
        #expect(arch.hiddenActivation == "gelu_pytorch_tanh")
        #expect(arch.partialRotaryFactor == 0.25)
        #expect(arch.fullRopeTheta == 1_000_000.0)
        #expect(arch.ropeTheta == 10_000.0)
    }

    /// ...and the same keys, absent, still take their documented defaults. This
    /// is the half that stops the strictness above from turning into "every key
    /// is now mandatory", which would reject configs that parse today.
    @Test func gemmaOptionalKeysStillDefaultWhenAbsent() throws {
        let arch = try loadGemma(dropping: ["tie_word_embeddings", "attention_k_eq_v",
                                            "hidden_activation", "rope_parameters"],
                                 tag: "gemma-absent-optionals")
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.attentionKEqV == false)
        #expect(arch.hiddenActivation == "gelu_pytorch_tanh")
        #expect(arch.partialRotaryFactor == 0.25)
        #expect(arch.fullRopeTheta == 1_000_000.0)
        #expect(arch.ropeTheta == 10_000.0)
    }

    /// A RoPE base stated as a boolean used to become the default silently —
    /// 1_000_000 for the full-attention arm — which is a wrong rotary base
    /// applied to every layer with nothing to catch it.
    @Test func gemmaConfigRejectsAWrongTypedRopeTheta() throws {
        #expect(throws: RepackError.self) {
            _ = try loadGemma(
                overriding: ["rope_parameters": [
                    "full_attention": ["rope_theta": true],
                    "sliding_attention": ["rope_theta": 10_000.0],
                ]],
                tag: "gemma-bool-theta")
        }
    }

    // MARK: - Ling

    /// The whole parse, pinned against the published config. Every number here
    /// was transcribed from `inclusionAI/Ling-mini-2.0`, so a mistake in the
    /// parser shows up as a mismatch rather than as agreement between two
    /// copies of the same error.
    @Test func bailingConfigParsesLingsPublishedValues() throws {
        let directory = temporaryRoot("ling-arch")
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try BailingSyntheticConfig.write(to: directory)

        let arch = try ArchInfo.load(configPath: path)

        // THE TRAP, pinned: `intermediateSize` is the shared-expert width and
        // must be 512. Ling's config also has `intermediate_size` = 5120, the
        // dense layer-0 width, and it belongs in the payload. Swap these two
        // and the shared-expert GEMV is sized ten times wrong with no error
        // anywhere.
        #expect(arch.intermediateSize == 512)
        guard case let .bailingMoeV2(extras) = arch.variant else {
            Issue.record("expected the bailingMoeV2 variant, got \(arch.variant)")
            return
        }
        #expect(extras.denseIntermediateSize == 5120)

        #expect(arch.hiddenSize == 2048)
        #expect(arch.moeIntermediateSize == 512)
        #expect(arch.numHeads == 16)
        #expect(arch.numKVHeads == 4)
        #expect(arch.headDim == 128)
        #expect(arch.vocabSize == 157_184)
        #expect(arch.numLayers == 20)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 8)
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.hiddenActivation == "silu")
        #expect(arch.partialRotaryFactor == 0.5)
        #expect(arch.fullRopeTheta == 600_000)
        // Not left at Gemma's 10_000 sliding-window fallback.
        #expect(arch.ropeTheta == 600_000)

        // No sliding/global split: the "full" head geometry is the head
        // geometry, and every layer is full attention. A mask that is empty or
        // shorter than `numLayers` is indexed by layer downstream.
        #expect(arch.numFullKVHeads == 4)
        #expect(arch.fullHeadDim == 128)
        #expect(arch.fullAttentionLayerMask == [UInt8](repeating: 1, count: 20))
        #expect(arch.slidingWindow == 0)
        #expect(arch.finalLogitSoftcap == 0.0)
        #expect(arch.attentionKEqV == false)

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

    /// Dispatch falls back to `architectures` exactly as the pre-download gate
    /// does, so a config missing `model_type` cannot take one route through the
    /// gate and another through the parse.
    @Test func bailingDispatchesOnArchitecturesWhenModelTypeIsAbsent() throws {
        let directory = temporaryRoot("ling-arch-only")
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try BailingSyntheticConfig.write(to: directory,
                                                    dropping: ["model_type"])

        let arch = try ArchInfo.load(configPath: path)
        #expect(arch.variant.family == .bailingMoeV2)
    }

    // MARK: - Anti-inheritance: one case per Gemma silent default

    /// Ling carries `hidden_act` — a different key *name* from Gemma's
    /// `hidden_activation`. A shared parse would find nothing, fall back to
    /// `"gelu_pytorch_tanh"` and report success.
    @Test func bailingConfigWithoutHiddenActThrowsRatherThanDefaultingToGemmasActivation() throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(dropping: "hidden_act", tag: "ling-no-act")
        }
    }

    /// Would otherwise silently become Gemma's 0.25 — half of Ling's rotary
    /// dimension, applied to every layer.
    @Test func bailingConfigWithoutPartialRotaryFactorThrows() throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(dropping: "partial_rotary_factor", tag: "ling-no-prf")
        }
    }

    /// Would otherwise silently become Gemma's 1_000_000 instead of Ling's
    /// 600_000.
    @Test func bailingConfigWithoutRopeThetaThrows() throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(dropping: "rope_theta", tag: "ling-no-theta")
        }
    }

    /// Ling publishes `tie_word_embeddings: false`; Gemma's branch defaults a
    /// missing key to `false` too, so the *value* would happen to be right. It
    /// still has to throw: a config that does not state it is not a config we
    /// have read, and the next architecture to want `true` would inherit a
    /// wrong answer with no test to catch it.
    @Test func bailingConfigWithoutTieWordEmbeddingsThrows() throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(dropping: "tie_word_embeddings", tag: "ling-no-tie")
        }
    }

    /// The trap, from the other side. `intermediate_size` (5120) is present and
    /// parses; only the shared-expert key is gone. Reading the wrong one would
    /// make this load succeed with `intermediateSize == 5120`.
    @Test func bailingConfigWithoutSharedExpertWidthThrowsInsteadOfReadingIntermediateSize() throws {
        let directory = temporaryRoot("ling-no-shared")
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try BailingSyntheticConfig.write(
            to: directory, dropping: ["moe_shared_expert_intermediate_size"])
        // Precondition of the test: the decoy key is still there.
        let config = try jsonObject(at: path)
        #expect((config["intermediate_size"] as? Int) == 5120)

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    // MARK: - Type strictness

    /// `JSONSerialization` returns JSON numbers and JSON booleans alike as
    /// `NSNumber`, and Swift's `as? Bool` succeeds for any NSNumber holding 0 or
    /// 1. So `use_qk_norm: 1` used to be read as `true` — a fact nobody stated,
    /// invented by the one component whose job is to not invent architecture
    /// facts. A config that says it in the wrong type has not been read.
    @Test func bailingConfigRejectsAnIntegerWhereABooleanIsMeant() throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(overriding: ["use_qk_norm": 1], tag: "ling-int-bool")
        }
    }

    /// The same confusion in the other direction: `moe_router_enable_expert_bias`
    /// is a Bool, and 0 is not `false` until someone decides it is.
    @Test func bailingConfigRejectsZeroWhereABooleanIsMeant() throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(overriding: ["moe_router_enable_expert_bias": 0],
                                tag: "ling-zero-bool")
        }
    }

    /// `as? Int` on an NSNumber truncates, so `num_hidden_layers: 20.5` would
    /// have produced a 20-layer model — and, worse, a 20-entry
    /// `fullAttentionLayerMask` derived from it, so nothing downstream could
    /// even notice the config disagreed.
    @Test func bailingConfigRejectsANonIntegerLayerCount() throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(overriding: ["num_hidden_layers": 20.5],
                                tag: "ling-fractional-layers")
        }
    }

    /// A number spelled as a string is not a number.
    @Test func bailingConfigRejectsAStringWhereANumberIsMeant() throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(overriding: ["hidden_size": "2048"], tag: "ling-string-int")
        }
    }

    /// And a boolean where the activation name belongs.
    @Test func bailingConfigRejectsANonStringActivation() throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(overriding: ["hidden_act": true], tag: "ling-bool-act")
        }
    }

    // MARK: - Out-of-range numbers must throw, not trap

    /// `config.json` is a downloaded file, and `Int(exactly:)` bounds the TYPE
    /// but not the VALUE. `num_hidden_layers: -1` reached
    /// `[UInt8](repeating: 1, count: numLayers)`, and that is a **trap**: the
    /// repacker aborted mid-parse instead of reporting an invalid config.
    ///
    /// Zero is in the table for a different reason — it does not trap, it
    /// yields an empty mask, and the runtime indexes that array by layer.
    @Test(arguments: [-1, 0, ArchInfo.maxPlausibleDimension + 1])
    func bailingConfigRejectsAnOutOfRangeLayerCount(_ layers: Int) throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(overriding: ["num_hidden_layers": layers],
                                tag: "ling-layers-\(layers)")
        }
    }

    /// The same for the widths. None of these traps on its own, but each
    /// propagates into the manifest and then into buffer sizing, and a
    /// negative width is not something the parse should be inventing an
    /// interpretation for.
    @Test(arguments: ["hidden_size", "moe_shared_expert_intermediate_size",
                      "moe_intermediate_size", "intermediate_size",
                      "num_attention_heads", "num_key_value_heads", "head_dim",
                      "vocab_size", "num_experts", "num_experts_per_tok",
                      "n_group", "topk_group", "num_shared_experts"])
    func bailingConfigRejectsANonPositiveDimension(_ key: String) throws {
        #expect(throws: RepackError.self) {
            _ = try loadBailing(overriding: [key: -1], tag: "ling-negative-\(key)")
        }
        #expect(throws: RepackError.self) {
            _ = try loadBailing(overriding: [key: 0], tag: "ling-zero-\(key)")
        }
    }

    /// `first_k_dense_replace` is the one count where zero is a real answer: it
    /// means the model has no dense prefix. Guarding it like the widths would
    /// reject a legitimate config, so it is guarded one step looser — and that
    /// looseness is pinned rather than left to be discovered.
    @Test func bailingConfigAcceptsZeroDenseLayersButNotANegativeCount() throws {
        let arch = try loadBailing(overriding: ["first_k_dense_replace": 0],
                                   tag: "ling-zero-dense")
        guard case let .bailingMoeV2(extras) = arch.variant else {
            Issue.record("expected the bailingMoeV2 variant, got \(arch.variant)")
            return
        }
        #expect(extras.firstKDenseReplace == 0)

        #expect(throws: RepackError.self) {
            _ = try loadBailing(overriding: ["first_k_dense_replace": -1],
                                tag: "ling-negative-dense")
        }
    }

    // MARK: - Unknown architectures

    /// An unsupported checkpoint must be refused at the parse, not adopted by
    /// whichever branch happens to tolerate its keys.
    ///
    /// The error has to *name* the type it refused. Merely throwing is not
    /// enough to prove the dispatcher rejected it: routing an unknown config
    /// into the Gemma branch also throws — "no text_config" — and that message
    /// sends the reader looking for a missing key in a config that was never
    /// Gemma's to begin with.
    @Test func unknownModelTypeThrowsAndNamesTheType() throws {
        let directory = temporaryRoot("unknown-arch")
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try BailingSyntheticConfig.write(
            to: directory,
            overriding: ["model_type": "qwen3_moe",
                         "architectures": ["Qwen3MoeForCausalLM"]])

        do {
            _ = try ArchInfo.load(configPath: path)
            Issue.record("expected an unsupported-architecture error")
        } catch let error as RepackError {
            #expect(String(describing: error).contains("qwen3_moe"))
        }
    }

    /// A config claiming nothing at all is refused too — previously this
    /// surfaced as "no text_config", which described the Gemma branch rather
    /// than the config.
    @Test func configWithNeitherModelTypeNorArchitecturesThrows() throws {
        let directory = temporaryRoot("nameless-arch")
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try BailingSyntheticConfig.write(
            to: directory, dropping: ["model_type", "architectures"])

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    // MARK: - Helpers

    /// The published `google/gemma-4-26b-a4b` text config, trimmed to the keys
    /// `loadGemma4` reads. Transcribed from the real thing, not generated from
    /// `ArchInfo` — a fixture derived from the parser under test would agree
    /// with any mistake in it.
    ///
    /// `layer_types` is the real 30-layer pattern: every sixth layer, starting
    /// at index 5, is full attention.
    private static var gemmaTextConfig: [String: Any] {
        var layerTypes = [String](repeating: "sliding_attention", count: 30)
        for i in stride(from: 5, to: 30, by: 6) { layerTypes[i] = "full_attention" }
        return [
            "hidden_size": 2816,
            "intermediate_size": 2112,
            "moe_intermediate_size": 704,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262_144,
            "num_hidden_layers": 30,
            "num_experts": 128,
            "top_k_experts": 8,
            "sliding_window": 1024,
            "final_logit_softcapping": 30.0,
            "rope_parameters": [
                "sliding_attention": ["rope_theta": 10_000.0, "rope_type": "default"],
                "full_attention": ["rope_theta": 1_000_000.0,
                                   "rope_type": "proportional",
                                   "partial_rotary_factor": 0.25],
            ],
            "layer_types": layerTypes,
            "tie_word_embeddings": true,
            "attention_k_eq_v": true,
            "hidden_activation": "gelu_pytorch_tanh",
        ]
    }

    @discardableResult
    private func writeGemmaConfig(to directory: String,
                                  dropping: [String] = [],
                                  overriding: [String: Any] = [:]) throws -> String {
        try FileManager.default.createDirectory(atPath: directory,
                                                withIntermediateDirectories: true)
        var textConfig = Self.gemmaTextConfig
        for key in dropping {
            precondition(textConfig[key] != nil,
                         "dropping \(key), which the fixture does not carry — "
                         + "the test would pass vacuously")
            textConfig.removeValue(forKey: key)
        }
        for (key, value) in overriding { textConfig[key] = value }
        let config: [String: Any] = [
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "text_config": textConfig,
        ]
        let path = (directory as NSString).appendingPathComponent("config.json")
        let data = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }

    private func loadGemma(dropping: [String] = [],
                           overriding: [String: Any] = [:],
                           tag: String) throws -> ArchInfo {
        let directory = temporaryRoot(tag)
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try writeGemmaConfig(to: directory,
                                        dropping: dropping,
                                        overriding: overriding)
        return try ArchInfo.load(configPath: path)
    }

    private func loadBailing(dropping key: String, tag: String) throws -> ArchInfo {
        let directory = temporaryRoot(tag)
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try BailingSyntheticConfig.write(to: directory, dropping: [key])
        return try ArchInfo.load(configPath: path)
    }

    private func loadBailing(overriding: [String: Any], tag: String) throws -> ArchInfo {
        let directory = temporaryRoot(tag)
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let path = try BailingSyntheticConfig.write(to: directory, overriding: overriding)
        return try ArchInfo.load(configPath: path)
    }

    private func jsonObject(at path: String) throws -> [String: Any] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    private func write(_ config: [String: Any], to path: String) throws {
        let data = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private func temporaryRoot(_ tag: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("archinfo-\(tag)-\(UUID().uuidString)")
    }
}
