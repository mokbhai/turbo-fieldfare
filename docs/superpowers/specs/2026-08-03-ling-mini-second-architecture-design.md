# Adding a second architecture: Ling-mini-2.0

Design for issue [#44](https://github.com/drumih/turbo-fieldfare/issues/44),
"implement other MOE family models".

Status: design approved in outline, not yet planned into tasks. Every file and
line reference below was read and independently verified against the tree at
`1a8ef5b`.

## Why Ling-mini-2.0 and not Qwen

The request was for a Qwen MoE. Reading the published configs rules that out:

- **There is no Qwen 3.8.** Qwen's current line is Qwen3.6 — `Qwen/Qwen3.6-35B-A3B`
  (MoE) and `Qwen/Qwen3.6-27B` (dense).
- **Qwen 3.5 and 3.6's MoE are hybrid linear-attention models.** `qwen3_5_moe`
  declares 40 layers of which **36 are `linear_attention`** — gated-DeltaNet style,
  with `linear_conv_kernel_dim`, `mamba_ssm_dtype`, and separate linear key/value
  heads — plus `attn_output_gate`, interleaved mRoPE, and a vision tower. This
  runtime's entire attention story (`Attention.swift`, `PrefillAttention.swift`,
  `KVCacheManager`) would apply to 4 of 40 layers. Supporting it means a Metal
  linear-attention/SSM kernel family and a recurrent state cache — a much larger
  project than adding an architecture, and one worth doing only after a second
  *standard-attention* model has proven the seams.
- **Qwen has no ~10B MoE.** The smallest official Qwen MoE is 30B-A3B.

`inclusionAI/Ling-mini-2.0` (`bailing_moe`) is the closest fit to "small MoE with
1–2B active" that still uses ordinary softmax attention: 16.25B total, ~1.4B
active, 20 layers, 256 experts top-8, one shared expert, GQA 16/4, published as
`mlx-community/Ling-mini-2.0-4bit` in the same affine 4-bit / group-64 / BF16-scale
format the repacker already consumes.

The streaming economics are better than Gemma's, which is the real argument:

| | Gemma 4 26B-A4B | Ling-mini-2.0 |
|---|---|---|
| Per-expert blob | 3,358,720 B (padded from 3,345,408) | **1,769,472 B — exactly 108 × 16 KB, zero padding** |
| Routed weights on disk | 12.90 GB (30 layers) | 8.61 GB (19 MoE layers) |
| **Streamed per token** | 30 × 8 × 3.36 MB = **806 MB** | 19 × 8 × 1.77 MB = **269 MB** |
| Resident (non-routed) | ~1.35 GB | **~0.54 GB** |

3× less SSD traffic per token and 40% of the resident footprint. The caveat is
that *miss count* doubles — 8-of-256 is half the selection density of 8-of-128 —
and pread latency rather than bandwidth dominates decode, so the 16-slot expert
cache default must not simply be inherited.

## Governing principle: additive, never parameterized

Every Gemma code path either moves verbatim into a named conformance or is not
touched. New behaviour arrives as siblings. Two facts force this:

1. **Metal is one compilation unit.** Editing a device function changes Gemma's
   numerics at runtime and the kernel tests would not catch it, because they
   compare against `MoeRef`, which still uses gelu. Ling's shaders go in a new
   `Sources/TurboFieldfare/Metal/Ling/ling.metal`, registered after `moe` and
   `prefill`, using function constants **90–99** — `fused.metal:18-24` already
   occupies 80–86, and a collision bricks the library at `MetalContext.swift:117`
   for Gemma too.
2. **`manifest.flags` rejects unknown keys** (`ManifestReader.swift:124-128`), so
   the architecture discriminator must ride in `arch`, not `flags`.

## Architecture: four seams

### 1. Arch description

`ArchFamily` / `ArchVariant` with `Gemma4Extras` and `BailingMoeV2Extras` payloads.

These must be **declared twice** — once beside `ArchInfo` in
`TurboFieldfareRepackCore`, once beside `ArchConfig` in `TurboFieldfare` — because
`Package.swift:23-36` gives the two targets no dependency edge. A shared module
would also perturb `TurboFieldfareAppCore`. Pin the copies with a cross-target
equality test.

- `ManifestArch` gets a hand-written `init(from:)` defaulting `family` to `gemma4`,
  so every shipped Gemma manifest decodes byte-identically.
- `ArchInfo.load` becomes a dispatcher on `model_type`. Ling's config is flat with
  no `text_config` (`ArchInfo.swift:32` is the first thing a Ling repack hits) and
  none of Gemma's six required keys. The bailing branch must **not** inherit the
  silent Gemma defaults at `ArchInfo.swift:47-60`.
- `intermediateSize` is documented and consumed as the **shared-expert** FFN width
  (`ModelTypes.swift:8`, `:77`), so Ling sets it to 512; the dense layer-0 width of
  5120 lives in `BailingMoeV2Extras.denseIntermediateSize` and needs its own
  manifest key, because `validateArch:202` cannot otherwise express it.
- `ManifestReader.validateQuant` hardcodes router == 8 bits (`:166`). Ling's router
  is 4-bit. Move the bit-width table onto `ArchConfig` **per-architecture** — a flat
  `[4, 8]` widening would let Gemma install with a 4-bit router.

### 2. Tensor naming

`protocol TensorNaming` owning every name-shaped decision in the repacker:
classify, routed-expert role, companion filter, resident order, quant-slot name.
`Gemma4Naming` holds the current bodies **verbatim**.

`RepackPlanner.classify:107` gates everything on the `language_model.` prefix and
throws `unknownTensorPrefix` otherwise, so every Ling tensor (`model.*`, `lm_head.*`)
throws before the routed-expert test is even reached.

**The fused `query_key_value` needs no repack work.** The split is along *rows*
(q 0–2047, k 2048–2559, v 2560–3071), so all three sub-views start on a row
boundary — exactly what the existing per-call `weightsOffset`/`scalesOffset`/
`biasesOffset` already express (`prefill.metal:690-692`, `tensorops.metal:69-73`,
`dequant_int4.metal:229-241`). `TensorView` has a public memberwise init
(`ModelTypes.swift:211-215`). The tensor stays fused on disk and is sliced into
three `TensorView`s in a new `Model+Ling.swift`. All byte offsets are even,
satisfying the 2-byte alignment preconditions at `DequantInt4GEMV.swift:61`.

**Dense layer 0 is real work.** `first_k_dense_replace: 1` means layer 0 has no
experts and no router. `RepackPlanner`'s `expertsPerLayer: 0` branch (`:197-205`)
looks like it already handles this, but it is **dead code** — `SyntheticSnapshot.swift:106-117`
gives every layer routed experts, so it has never executed, and it breaks
downstream in `PackedExpertsLayout`, `Model.openLayerLocked`,
`Model.validateTrustedReceiptLayerLayout`, `ManifestReader.validate:153`,
`VerifiedInstallTool` and `GTurboLayoutValidator`. Note `VerifiedInstallTool:150`
(`layers.count == numLayers`) fires *before* the checks the runtime loader makes,
so verify-install is strictly stricter than `Model.load`: fixing one without the
other ships an install that verifies but will not load.

### 3. Runtime execution

`LingForwardRunner` behind the existing `LogitProducer` /
`ContinuableLogitProducer` / `ChunkedPrefillRunner` protocols, reached through a
factory. Gemma's `RealForwardRunner` is not refactored.

Decode per layer (D=2048, headDim=128, 16Q/4KV, F_moe=F_shared=512, E=256, top-8):

```
1  RMSNorm(hidden, input_layernorm)     layer 0 only; layers 1-19 fuse it into
                                        the previous layer's ling_add_rmsnorm
2  FusedQKVGEMV over the three sliced views -> K/V written straight to kv slots
3  ling_qk_epilogue   per-head norm on Q(16)/K(4) with the shared [128] weight,
                      partial NeoX RoPE (rotary_dim 64, theta 600000), NO V norm
4  Attention.encodeFull(128, 16, 4, scale = 1/sqrt(128))   <- unmodified kernel
5  DequantInt4GEMV o_proj
6  ling_add_rmsnorm(hidden += attn, post_attention_layernorm) -> n2
7  SharedExpertInt4(activation: .silu) on n2 -> sharedOut
8  LingRouter: int4 GEMV -> fp32 logits[256], then the gate
9  LingRoutedMoE phase1-silu + phase2 with residual: sharedOut
```

Step 9 is free: `moe_phase2_down_reduce_k8` (`moe.metal:446`) already computes
`residual + sum(w * down(acts))` and Gemma wastes the slot binding zeros, so
`out = h + Shared + Routed` costs no extra dispatch.

The router gate, transcribed from `BailingMoeV2Gate.forward`:

```
scores            = sigmoid(logits)              # fp32
scores_for_routing = scores + expert_bias        # bias affects SELECTION ONLY
group_scores      = view(8 groups of 32).topk(2).sum(-1)
keep              = topk(group_scores, 4)        # 4 of 8 groups
topk_idx          = topk(masked scores_for_routing, 8)
w                 = gather(scores, topk_idx)     # UNBIASED sigmoid scores
w                 = w / (w.sum() + 1e-20) * 2.5
```

### 4. Chat framing

`protocol ChatFraming`; Gemma's string builder moves verbatim into
`Gemma4Framing`. Ling renders its shipped `chat_template.jinja` **directly to
token IDs** rather than round-tripping through a decoded string. Its framing is
`<role>SYSTEM</role>…<|role_end|>` / `<role>HUMAN</role>` / `<role>ASSISTANT</role>`,
with `add_bos_token: false` (no BOS at all, unlike Gemma), eos `<|role_end|>`, and
`detailed thinking off` hardcoded in the template.

`Tokenizer.swift:59-65` currently falls back to the pinned Gemma Hub repo whenever
a sidecar is missing — a 262k-vocab tokenizer against a 157k-vocab model, no
exception, out-of-range embedding lookups. **This must close before any Ling
catalog entry ships.**

## Staging

Each milestone ends in something verifiable.

- **M0 — Freeze Gemma.** Pin `canonicalFingerprint`, `residentIndexSha256` and the
  scalar-copy count as golden values; add a test that the combined Metal library
  compiles and every existing Gemma pipeline resolves. Must precede everything.
- **M1 — Arch plumbing, no execution.** Families, variants, `ArchInfo` dispatch,
  variant-aware `validateArch`/`validateQuant`. `ArchConfig.lingMini2_0_4bit`
  defined but **not** in `supported`. *Verifiable: shipped Gemma manifests still
  load byte-identically; a hand-written Ling manifest validates.*
- **M2 — Dense layer 0 + tensor naming, end to end on synthetics.** *Verifiable:
  a Ling synthetic repacks, verifies and loads; the `expertsPerLayer: 0` branch
  stops being dead code; Gemma's fingerprint is unchanged.*
- **M3 — References and kernels.** `BailingRouterRef`, `MoeRef.runFFNSilu`,
  `RopeRef.applyPartialNeox`, then `ling.metal`. *Verifiable: every Ling kernel
  matches its FP32 reference, with **exact** expert-index agreement.*
- **M4 — Decode-only Ling. First Ling token.** Prefill runs as a `produce()` loop
  reporting `PrefillExecutionDiagnostics.unsupported` so slow numbers cannot be
  mistaken for real chunked prefill.
- **M5 — Chunked prefill.** *Verifiable: chunked and decode-loop paths agree on
  first-token logits.*
- **M6 — Tokenizer and catalog. First usable chat.** `ArchConfig.supported` gains
  Ling **here**, with kernels in place.
- **M7 — Correctness gate.** Env-gated mlx-lm parity fixture with per-layer expert
  IDs so a routing divergence localizes.
- **M8 — Optional performance.** headDim-128 TensorOps prefill attention is the
  largest lever, since Ling has no SWA layers and all 20 pay full quadratic cost.

## Risks

Ranked, known-hard first.

1. **Partial-RoPE convention.** Three Gemma implementations (`fused.metal:54-61`,
   `rope.metal:32`, `prefill.metal:754-774`) and **both** reference functions
   encode `(i, i + head_dim/2)` with divisor `head_dim`. Ling needs `(i, i + 32)`
   with divisor 64 and dims [64,128) untouched. Reaching for `applyNeox` makes the
   reference wrong the same way the kernel is wrong, so the test passes and the
   model generates fluent nonsense that degrades with context. `fused.metal` is the
   copy most likely to be reached for. Write `applyPartialNeox` first, cross-check
   against a brute-force formulation, and validate against mlx-lm before anything
   else.
2. **Gemma byte-identity through the `TensorNaming` refactor.** It rewrites the code
   producing resident order, hence every `fileOffset`, the index SHA, and
   `canonicalFingerprint`. Drift silently invalidates in-flight Gemma resume
   checkpoints. Mitigated only by M0.
3. **`versionMinor` bump invalidates resume checkpoints.**
   `RangeCopyPlanner.canonicalFingerprint:250-252` hashes it; users mid-download
   lose ~14 GB. Decide before M1 whether to exclude minor from the fingerprint.
4. **Tolerance hiding a routing bug.** 8 of 256 scaled ×2.5 — one wrong expert can
   pass `fp16ChainedReduction`. Every routing test must assert indices with
   **exact** equality.
5. **Group-limited top-k tie semantics.** sigmoid compresses 256 candidates into a
   narrow band and group scores sum only two elements, so near-ties are far more
   common than in Gemma's dense top-8. Pin lower-index-wins at both stages and test
   a cross-group and a within-group tie deliberately. Use `precise::exp`, never
   `fast::exp` — Gemma's `fast::exp` is safe only because softmax is shift-invariant.
6. **`expert_bias` must affect selection only.** Gather the weights from the biased
   scores and nothing crashes; every token still routes plausibly and only quality
   degrades. Needs a dedicated test where the bias is large enough to change
   selection while the returned weight is the unbiased score.
7. **The V-norm trap.** `PrefillQKVEpilogue.swift:69-78` unconditionally applies
   Gemma's weightless per-head `v_norm` to V. BailingMoeV2 has only query/key
   layernorms. Reusing that epilogue silently RMS-normalizes every V row written
   into the KV cache — same failure class as (6).
8. **All-full attention.** No SWA layers means eager KV allocation at maxContext:
   40 KB/token, 1.31 GB at 32k, with no ring to fall back on. Context must be
   capped by policy.

## Must be measured, not decided

- The curated entry's `revision`, `sourceIndexSHA256`, `approximateDownloadBytes`
  and `installedBytes`. A guessed fingerprint makes `ModelTrustPolicy.decide`
  return `.curatedFingerprintMismatch` and hard-blocks every install.
  `installedBytes` must be read off a **completed** repack.
- Whether `mlx-community/Ling-mini-2.0-4bit` ships `chat_template.jinja` as a
  separate file. `ServerInference.swift:121-128` refuses to start without it while
  the repacker fetches it as optional.
- Whether the config carries a top-level `quantization` object —
  `IndexLoader.load:53-55` throws without one.
- Real `layout.json` size. 19 × 256 = 4,864 expert entries × 9 sub-objects,
  pretty-printed with sorted keys, is roughly 8.5 MB against a 16 MB cap, and it is
  SHA-256'd eagerly on every load (`Model.swift:368`).
- Router precision. `router_dtype` is fp32 but activations entering the GEMV are
  fp16 post-RMSNorm; at 256 experts a small logit bias can flip group selection.
