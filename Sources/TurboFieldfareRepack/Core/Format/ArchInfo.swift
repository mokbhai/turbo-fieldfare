import Foundation

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`. Indexed by layer.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String
    /// Which architecture this describes, carrying the facts only that family
    /// has. Everything above is a field both families genuinely populate; a
    /// fact that exists for one family only belongs in the payload, never as a
    /// fabricated value on a core field.
    let variant: ArchVariant

    /// The families the repacker can actually lay out.
    ///
    /// **Parsing a family and repacking it are different capabilities.**
    /// `load` gained a BailingMoeV2 branch, and gaining it silently removed a
    /// fail-fast: `config.json` used to be rejected with "no text_config" the
    /// moment a Ling repack was attempted, and now it parses cleanly and the
    /// run continues — through the shard-header downloads, into a planner whose
    /// tensor classification, resident ordering template and expert bundling
    /// are all written against Gemma's names. The eventual failure is
    /// `unknownTensorPrefix` on whichever Ling tensor happens to be visited
    /// first, tens of megabytes later, and it describes the tensor rather than
    /// the decision.
    ///
    /// Keyed on the family, and separate from the runtime's
    /// `ArchConfig.executableFamilies`, because they are genuinely different
    /// facts: one is "this binary has Metal kernels", the other is "this
    /// planner knows this checkpoint's tensor layout". Ling will very plausibly
    /// gain one before the other.
    static let repackableFamilies: Set<ArchFamily> = [.gemma4]

    /// Refuse an architecture the planner has no layout for, naming the family
    /// and the decision rather than the first tensor that did not match.
    func validateRepackable() throws {
        guard Self.repackableFamilies.contains(variant.family) else {
            throw RepackError.configurationInvalid(
                detail: "this build cannot repack architecture "
                      + "\(variant.family.rawValue): no layout planner for it")
        }
    }

    /// Ceiling for any count or width read out of a `config.json`. Far above
    /// anything published — the largest number in either supported checkpoint
    /// is Ling's 157,184-entry vocabulary — and present only so a corrupt
    /// value cannot be turned into an allocation before anything compares it.
    static let maxPlausibleDimension = 1_000_000

    /// Parse `config.json`. The top-level `model_type` picks the branch: the
    /// two supported checkpoints do not share a config shape, let alone a key
    /// vocabulary, so there is no common parse to factor out.
    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        // The TOP level, deliberately: Gemma's `text_config.model_type` is
        // `gemma4_text`, so dispatching on the nested value would miss.
        // `?? architectures.first` mirrors `ArchPreflight.evaluate`, so the
        // pre-download gate and this parse agree on what a config claims to be.
        guard let identifier = (root["model_type"] as? String)
                ?? (root["architectures"] as? [String])?.first else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "no model_type or architectures")
        }
        switch identifier {
        case "gemma4", "Gemma4ForConditionalGeneration":
            return try loadGemma4(root: root, configPath: configPath)
        case "bailing_moe", "BailingMoeV2ForCausalLM":
            return try loadBailingMoeV2(root: root, configPath: configPath)
        default:
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "unsupported model_type \(identifier)")
        }
    }

    /// Gemma's parse, over the SAME strict accessors the BailingMoeV2 parse
    /// uses. It used to have its own looser pair:
    ///
    ///     guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue
    ///
    /// which accepted a JSON boolean as 0/1, truncated `20.5` to 20, and put no
    /// bound on the value at all — so `num_hidden_layers: -1` reached
    /// `[UInt8](repeating:count:)` in the planner and TRAPPED the process. That
    /// asymmetry was not a decision about Gemma; it is just where the hardening
    /// stopped. The keys, their defaults and every value a valid Gemma config
    /// produces are unchanged — `gemmaConfigParsesThroughTheGemmaBranch` and the
    /// frozen-layout goldens pin that.
    private static func loadGemma4(root: [String: Any],
                                   configPath: String) throws -> ArchInfo {
        guard let textConfig = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        let tc = ConfigScope(values: textConfig, path: configPath)

        // Present-but-wrong-type is an error; absent keeps the historical
        // default. Gemma configs in the wild omit some of these, and this parse
        // predates the manifest that would have caught a wrong one, so the
        // defaults stay — what changes is that a value stated in the wrong type
        // is no longer coerced into one.
        let layerTypes = try tc.optionalStringArray("layer_types") ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = try tc.scopeOrEmpty("rope_parameters")
        let ropeFull = try rope.scopeOrEmpty("full_attention")
        let ropeSWA = try rope.scopeOrEmpty("sliding_attention")
        let prf = try ropeFull.optionalDouble("partial_rotary_factor") ?? 0.25
        let fullTheta = try ropeFull.optionalDouble("rope_theta") ?? 1_000_000.0
        let swaTheta = try ropeSWA.optionalDouble("rope_theta") ?? 10_000.0
        let kEqV = try tc.optionalBool("attention_k_eq_v") ?? false
        let tie = try tc.optionalBool("tie_word_embeddings") ?? false
        let act = try tc.optionalString("hidden_activation") ?? "gelu_pytorch_tanh"
        return ArchInfo(
            hiddenSize: try tc.count("hidden_size"),
            intermediateSize: try tc.count("intermediate_size"),
            moeIntermediateSize: try tc.count("moe_intermediate_size"),
            numHeads: try tc.count("num_attention_heads"),
            numKVHeads: try tc.count("num_key_value_heads"),
            numFullKVHeads: try tc.count("num_global_key_value_heads"),
            headDim: try tc.count("head_dim"),
            fullHeadDim: try tc.count("global_head_dim"),
            vocabSize: try tc.count("vocab_size"),
            // `nonNegativeCount`, like Ling's: zero is a real answer here and
            // means "no sliding window", which is what a fully-global Gemma
            // variant would say. Every other dimension is a count of something
            // that must exist.
            slidingWindow: try tc.nonNegativeCount("sliding_window"),
            // Not bounded: a softcap is a magnitude, not a count, and nothing
            // allocates from it.
            finalLogitSoftcap: try tc.double("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try tc.count("num_hidden_layers"),
            numExperts: try tc.count("num_experts"),
            topKExperts: try tc.count("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            variant: .gemma4)
    }

    /// `inclusionAI/Ling-mini-2.0`, consumed as `mlx-community/Ling-mini-2.0-4bit`.
    ///
    /// Ling's `config.json` is FLAT — there is no `text_config` — and it shares
    /// none of the six keys `loadGemma4` requires, so this constructs all 22
    /// fields fresh rather than adjusting a Gemma result.
    ///
    /// Every key Ling actually carries is read with a throwing, type-strict
    /// accessor, and none of the Gemma branch's `??` fallbacks are reachable
    /// from here. That is the whole point of the separation: `hidden_act` vs
    /// Gemma's `hidden_activation` is a different key *name*, so a shared parse
    /// would return nil, fall back to `"gelu_pytorch_tanh"`, and report no
    /// error. Nothing downstream would catch it either — `validateArch` compares
    /// the manifest against the hand-written `ArchConfig`, so if both are wrong
    /// the same way the model installs and runs the wrong activation forever.
    private static func loadBailingMoeV2(root: [String: Any],
                                         configPath: String) throws -> ArchInfo {
        let c = ConfigScope(values: root, path: configPath)
        func count(_ k: String) throws -> Int { try c.count(k) }
        func nonNegativeCount(_ k: String) throws -> Int { try c.nonNegativeCount(k) }
        func d(_ k: String) throws -> Double { try c.double(k) }
        func b(_ k: String) throws -> Bool { try c.bool(k) }
        func s(_ k: String) throws -> String { try c.string(k) }

        let extras = BailingMoeV2Extras(
            // THE TRAP: Ling's `intermediate_size` is the DENSE layer-0 FFN
            // width (5120), not the shared-expert width. This line is the only
            // place the key may be read; `intermediateSize` below reads
            // `moe_shared_expert_intermediate_size` instead. Swapping them
            // sizes the shared-expert GEMV ten times wrong, silently, because
            // both keys exist and both parse.
            denseIntermediateSize: try count("intermediate_size"),
            firstKDenseReplace: try nonNegativeCount("first_k_dense_replace"),
            numSharedExperts: try count("num_shared_experts"),
            nGroup: try count("n_group"),
            topkGroup: try count("topk_group"),
            routedScalingFactor: try d("routed_scaling_factor"),
            normTopkProb: try b("norm_topk_prob"),
            scoreFunction: try s("score_function"),
            routerEnableExpertBias: try b("moe_router_enable_expert_bias"),
            useQKNorm: try b("use_qk_norm"))

        // `count`, not `i`: the next line turns this number into an allocation,
        // and `[UInt8](repeating:count:)` traps rather than throws on a
        // negative one.
        let numLayers = try count("num_hidden_layers")
        // Ling has no `layer_types` and no sliding window: every layer is full
        // attention. Derived from `num_hidden_layers`, never defaulted — the
        // Gemma branch's `?? []` would yield an EMPTY mask, and the runtime
        // indexes this array by layer.
        let mask = [UInt8](repeating: 1, count: numLayers)
        // One RoPE base for every layer. `ropeTheta` (Gemma's sliding-window
        // theta) gets the same number rather than a fallback, so a future
        // reader of the non-full field sees Ling's value and not 10_000.
        let theta = try d("rope_theta")
        let headDim = try count("head_dim")
        let numKVHeads = try count("num_key_value_heads")

        return ArchInfo(
            hiddenSize: try count("hidden_size"),
            // SHARED-EXPERT FFN width, which is what this field means and how
            // the runtime consumes it. See the trap note above.
            intermediateSize: try count("moe_shared_expert_intermediate_size"),
            moeIntermediateSize: try count("moe_intermediate_size"),
            numHeads: try count("num_attention_heads"),
            numKVHeads: numKVHeads,
            // Ling draws no sliding/global distinction, so the "full" head
            // geometry *is* the head geometry. Derived from the all-full mask
            // above, not invented.
            numFullKVHeads: numKVHeads,
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try count("vocab_size"),
            // Concepts Ling does not have. 0 means "none" — a definite value,
            // and the runtime's `validateArch` compares it for Ling like every
            // other core field, so it must be written the same way here every
            // time or a repacked manifest stops loading. Note for whoever wires
            // `finalLogitSoftcap` through in a later milestone — the sampling
            // kernel computes `softcap * tanh(z / softcap)` with no zero
            // guard, so 0.0 must be treated as "disabled" at the call site.
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: try d("partial_rotary_factor"),
            numLayers: numLayers,
            numExperts: try count("num_experts"),
            topKExperts: try count("num_experts_per_tok"),
            tieWordEmbeddings: try b("tie_word_embeddings"),
            // Ling's K and V are distinct projections. The flag carries no
            // safety on its own — the Gemma runner derives V from K for any
            // full-attention layer without consulting it — which is why Ling
            // needs its own forward path, not a false here.
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: try s("hidden_act"),
            variant: .bailingMoeV2(extras))
    }
}

/// One JSON object out of a `config.json`, read with accessors that check what a
/// value **is** rather than what it can be coerced into.
///
/// **Shared by both parsers on purpose.** The two branches read disjoint key
/// vocabularies out of differently-shaped files, so there is no common *parse*
/// to factor out — but there is no reason for them to disagree about what
/// counts as a number, and while they did, the same corrupt value was reported
/// on one branch and silently adopted on the other.
///
/// The problem being solved: `JSONSerialization` returns JSON numbers AND JSON
/// booleans as `NSNumber`, and Swift's bridging casts blur them. `as? Bool`
/// succeeds for any NSNumber holding 0 or 1, so `use_qk_norm: 1` satisfies a
/// Bool accessor; `as? Int` and `.intValue` truncate, so `num_hidden_layers:
/// 20.5` becomes 20 layers. A config that states a value in the wrong type has
/// not been read, and guessing at it is how a wrong architecture number gets
/// *invented* here rather than reported — and nothing downstream catches an
/// invented one, because `validateArch` only compares the manifest against a
/// hand-written baseline.
private struct ConfigScope {
    let values: [String: Any]
    let path: String

    private func invalid(_ detail: String) -> RepackError {
        RepackError.configJsonInvalid(path: path, detail: detail)
    }

    private func required(_ k: String) throws -> Any {
        guard let raw = values[k] else { throw invalid("missing \(k)") }
        return raw
    }

    /// A JSON number, explicitly not a JSON boolean.
    private func number(_ k: String) throws -> NSNumber {
        let raw = try required(k)
        guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else {
            throw invalid("\(k) is not a number")
        }
        return n
    }

    /// An integer, exactly: `Int(exactly:)` refuses `20.5` rather than
    /// truncating it.
    func int(_ k: String) throws -> Int {
        guard let v = Int(exactly: try number(k).doubleValue) else {
            throw invalid("\(k) is not an integer")
        }
        return v
    }

    /// A count of something that must exist.
    ///
    /// `int` bounds the TYPE but not the VALUE, and `config.json` is a file this
    /// process did not write. A count is not a free integer: `num_hidden_layers:
    /// -1` reaches `[UInt8](repeating: 1, count:)`, and that is a TRAP, not a
    /// throw, so a corrupt config aborted the process instead of being reported.
    /// Every dimension goes through here for the same reason — a negative or
    /// absurd width propagates into the manifest and then into buffer sizing.
    ///
    /// The ceiling is deliberately far above any published model (the largest
    /// number in either supported checkpoint is Ling's 157k vocabulary) and
    /// exists only so a corrupt number cannot be turned into an allocation.
    func count(_ k: String) throws -> Int {
        let v = try int(k)
        guard v > 0, v <= ArchInfo.maxPlausibleDimension else {
            throw invalid("\(k) is out of range: \(v)")
        }
        return v
    }

    /// A count where zero is a real answer — `first_k_dense_replace: 0` means no
    /// dense prefix, `sliding_window: 0` means no sliding window — but a
    /// negative one is still corrupt.
    func nonNegativeCount(_ k: String) throws -> Int {
        let v = try int(k)
        guard v >= 0, v <= ArchInfo.maxPlausibleDimension else {
            throw invalid("\(k) is out of range: \(v)")
        }
        return v
    }

    /// An integer literal is a fine spelling of a Double — Ling writes
    /// `"rope_theta": 600000` — so only the boolean confusion is excluded.
    func double(_ k: String) throws -> Double {
        try number(k).doubleValue
    }

    func bool(_ k: String) throws -> Bool {
        let raw = try required(k)
        guard let n = raw as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else {
            throw invalid("\(k) is not a boolean")
        }
        return n.boolValue
    }

    func string(_ k: String) throws -> String {
        guard let v = try required(k) as? String else {
            throw invalid("\(k) is not a string")
        }
        return v
    }

    // MARK: - Optional readers
    //
    // `nil` when the key is ABSENT; a throw when it is present in the wrong
    // type. The Gemma branch has historical defaults for keys some configs omit,
    // and those stay — but "absent, so use the default" and "stated as something
    // else, so coerce it" are different situations, and only the first is a
    // decision anyone made.

    /// The value at `k`, treating an explicit JSON `null` as absent.
    ///
    /// `JSONSerialization` decodes `null` to `NSNull()`, which is not `nil`, so
    /// a bare `values[k] == nil` test would send `"key": null` to the strict
    /// reader and throw. Upstream configs use `null` to mean "unset" — the same
    /// thing as omitting the key — so it takes the documented default instead.
    ///
    /// This is deliberately the opposite of how `manifest.json` treats an
    /// explicit null in its `family` key. That file is written by this package
    /// and read back by it, so a null there is malformed and is rejected; a
    /// `config.json` is someone else's file and follows their convention.
    private func presentValue(_ k: String) -> Any? {
        let raw = values[k]
        return raw is NSNull ? nil : raw
    }

    func optionalDouble(_ k: String) throws -> Double? {
        presentValue(k) == nil ? nil : try double(k)
    }

    func optionalBool(_ k: String) throws -> Bool? {
        presentValue(k) == nil ? nil : try bool(k)
    }

    func optionalString(_ k: String) throws -> String? {
        presentValue(k) == nil ? nil : try string(k)
    }

    func optionalStringArray(_ k: String) throws -> [String]? {
        guard let raw = presentValue(k) else { return nil }
        guard let v = raw as? [String] else {
            throw invalid("\(k) is not an array of strings")
        }
        return v
    }

    /// A nested object, or an EMPTY scope when the key is absent — which keeps
    /// the Gemma branch's `?? [:]` behaviour, so a config with no
    /// `rope_parameters` still falls through to the per-key defaults instead of
    /// failing. A key that is present but is not an object still throws.
    func scopeOrEmpty(_ k: String) throws -> ConfigScope {
        guard let raw = presentValue(k) else { return ConfigScope(values: [:], path: path) }
        guard let v = raw as? [String: Any] else {
            throw invalid("\(k) is not an object")
        }
        return ConfigScope(values: v, path: path)
    }
}
