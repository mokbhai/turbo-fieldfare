import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct RangeCopyPlannerTests {

    /// Frozen Gemma byte layout. Every constant below was measured, not chosen.
    ///
    /// WHAT THESE ARE
    /// The plan the repacker derives for `SyntheticSnapshot`'s Gemma model:
    /// which source bytes land at which offset in `model_weights.bin` and the
    /// per-layer expert blobs, in which order, coalesced into which ranges.
    /// `canonicalFingerprint` is the SHA of exactly that, and
    /// `RemoteStreamingRepacker` refuses to resume an interrupted install whose
    /// checkpoint carries a different one — it throws `installStateIncompatible`
    /// and throws away roughly 14 GB of already-downloaded bytes.
    ///
    /// WHY THEY ARE PINNED
    /// Nothing else in the suite notices if the *layout* moves. Every existing
    /// test either compares two freshly built plans against each other or checks
    /// a plan's internal consistency, so a refactor that renames tensors,
    /// reorders `RepackPlanner.slotRank`, or changes padding produces a
    /// self-consistent — and completely different — on-disk file, silently. This
    /// test is the only thing standing between such a refactor and every user
    /// mid-download.
    ///
    /// HOW TO REGENERATE
    /// Print the corresponding field of `GemmaFrozenPlan.withPlans`'s
    /// `rangePlan` / `repackPlan` and copy the value in. That is the mechanical
    /// part, and it is not the point.
    ///
    /// WHEN CHANGING THEM IS LEGITIMATE — READ THIS BEFORE YOU EDIT A LITERAL
    /// Only when the `.gturbo` **format itself** deliberately changed: a new
    /// `GTurboJSON.versionMajor`/`versionMinor`, a new index encoding, a
    /// different alignment, a deliberate change to resident ordering. In that
    /// case the format change is the work, updating these numbers is the
    /// paperwork, and the pull request must say which in-flight installs it
    /// invalidates and how they are migrated or discarded.
    ///
    /// Never update a constant to make this test pass. A red assertion here does
    /// not mean the test is stale; it means the code under it just rewrote
    /// Gemma's on-disk bytes. If that was not the intent of your change, the
    /// change is wrong. Re-deriving the golden from the new behaviour converts a
    /// caught regression into a shipped one, which is strictly worse than having
    /// no test at all, because the green suite now vouches for it.
    ///
    /// Note the blast radius is wider than the layout: `canonicalFingerprint`
    /// also hashes `GTurboJSON.versionMinor`, so a version bump trips this test
    /// too. That is intended — a minor bump invalidates resume checkpoints just
    /// as thoroughly as a layout change does.
    ///
    /// SCOPE. `SyntheticSnapshot` is 2 layers, 2 experts and a 512-token vocab;
    /// it pins the *transform*, not real Gemma's bytes. It has one sliding and
    /// one full-attention layer, so it does not exercise the real 25/5 pattern.
    /// A naming change that only misbehaves at scale can still pass here. This
    /// test also plans at 4096 bytes per range, which no install uses; the
    /// production chunk size is pinned by the sibling test below.
    @Test func gemmaByteLayoutIsFrozen() throws {
        try GemmaFrozenPlan.withPlans(
            rangeChunkBytes: GemmaFrozenPlan.smallChunkBytes
        ) { repackPlan, rangePlan in
            #expect(
                rangePlan.canonicalFingerprint
                    == "3c9aa53735878e50b166eacc2a0d2748eabfd5f9666a1f4dacc003c305cd4fa2",
                """
                Gemma's remote install plan fingerprint moved. Resuming an \
                in-flight Gemma download is now impossible: the checkpoint \
                carries the old fingerprint and RemoteStreamingRepacker will \
                discard it. Do not update this constant unless the .gturbo \
                format changed on purpose.
                """)
            #expect(
                rangePlan.residentIndexSha256
                    == "d9765a92bc632e878ae6202047184a777a8bcbf05baf2273a4725d8ccc924a70",
                """
                The resident index bytes changed — tensor names, shapes or \
                offsets moved inside model_weights.bin.
                """)

            // Shape counts alongside the hashes, so a failure says what moved
            // instead of only that something did.
            #expect(repackPlan.resident.entries.count == 41)
            #expect(rangePlan.scalarCopies.count == 109)
            #expect(rangePlan.coalescedCopies.count == 86)
            #expect(rangePlan.remoteBytesToDownload == 300_204)
            #expect(rangePlan.remoteGapBytesDownloaded == 0)

            #expect(repackPlan.resident.indexSize == 16_384)
            #expect(repackPlan.resident.residentSize == 244_908)
            #expect(repackPlan.resident.totalSize == 261_292)

            let outputs = rangePlan.expectedOutputs
                .map { "\($0.relativePath)=\($0.size)" }
            #expect(outputs == [
                "model_weights.bin=261292",
                "packed_experts/layer_00.bin=32768",
                "packed_experts/layer_01.bin=32768",
            ])
        }
    }

    /// The same freeze, at the chunk size production actually installs with.
    ///
    /// `canonicalFingerprint` hashes `rangeChunkBytes`, so the test above pins
    /// the fingerprint of a configuration no user ever runs. Changing
    /// `RemoteChunkPolicy.defaultBytes` — say 64 MiB to 32 MiB, a plausible
    /// tuning tweak — rewrites the fingerprint of every real install and
    /// therefore discards every in-flight resume checkpoint, while leaving the
    /// 4096 golden perfectly green. This test is what turns that edit red.
    ///
    /// Both pins are kept because neither covers the other. At 4096 the whole
    /// snapshot is split and coalesced, so the ordering of `splitLargeCopies`
    /// and `coalesce` is under test but the production constant is not in the
    /// hash. At 64 MiB the snapshot collapses into a single range — the count
    /// below says so — so the split path is untested but the constant is
    /// pinned. Everything else in the layout is pinned by both.
    ///
    /// Everything the test above says about *not* editing a constant to make a
    /// failure go away applies here verbatim, with one extra wrinkle: this
    /// fingerprint also moves if someone changes `RemoteChunkPolicy`. That is
    /// not a licence to re-measure it — it is the alarm doing its job, and the
    /// change needs the same in-flight-install story as a format change.
    @Test func gemmaByteLayoutIsFrozenAtTheProductionChunkSize() throws {
        try GemmaFrozenPlan.withPlans(
            rangeChunkBytes: GemmaFrozenPlan.productionChunkBytes
        ) { _, rangePlan in
            #expect(
                rangePlan.canonicalFingerprint
                    == "cf96875e3a45066bb89d27580195248fe8282b436ea8c82c3eaf5906191c8fcc",
                """
                The fingerprint of a real Gemma install moved. Either the \
                layout changed, or RemoteChunkPolicy.defaultBytes did — both \
                strand every in-flight download at its current byte. Check \
                which by looking at gemmaByteLayoutIsFrozen: if that one is \
                still green, the chunk policy is what moved.
                """)

            // A single range for the whole shard, fanned out to all three
            // output files, against 86 at 4096 bytes. This is the measurement
            // behind "the split path is untested here": one range cannot be
            // ordered wrongly. The download volume is identical either way —
            // the copies are contiguous in the source, so coalescing them
            // wholesale picks up no gap bytes.
            #expect(rangePlan.coalescedCopies.count == 1)
            #expect(rangePlan.remoteBytesToDownload == 300_204)
            #expect(rangePlan.remoteGapBytesDownloaded == 0)
        }
    }

    @Test func canonicalFingerprintDoesNotDependOnAbsoluteOutputRoot() throws {
        let snapshotDirectory = temporaryRoot("snapshot")
        let firstOutput = temporaryRoot("first")
        let secondOutput = temporaryRoot("second")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDirectory)
            try? FileManager.default.removeItem(atPath: firstOutput)
            try? FileManager.default.removeItem(atPath: secondOutput)
        }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDirectory,
            seed: 0x1020_3040)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDirectory)
        let arch = try ArchInfo.load(
            configPath: (snapshotDirectory as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let firstPlan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: firstOutput)
        let secondPlan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: secondOutput)

        let first = try RangeCopyPlanner.plan(
            repackPlan: firstPlan,
            rangeChunkBytes: 4096)
        let second = try RangeCopyPlanner.plan(
            repackPlan: secondPlan,
            rangeChunkBytes: 4096)

        #expect(first.canonicalFingerprint == second.canonicalFingerprint)
        #expect(first.coalescedCopies.map(\.id) == second.coalescedCopies.map(\.id))
    }

    @Test func overlappingDestinationIntervalsAreRejected() throws {
        let root = temporaryRoot("overlap")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("file.bin")
        let copies = [
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 0,
                size: 10,
                destinationPath: output,
                destinationOffset: 0),
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 20,
                size: 10,
                destinationPath: output,
                destinationOffset: 9),
        ]

        #expect(throws: RepackError.self) {
            try RangeCopyPlanner.validateDestinationIntervals(
                copies,
                outputRoot: root)
        }
    }

    @Test func normalizedRelativePathRejectsEscape() throws {
        let root = temporaryRoot("escape")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let outside = (root as NSString).deletingLastPathComponent
            + "/outside.bin"

        #expect(throws: RepackError.self) {
            _ = try RangeCopyPlanner.normalizedRelativePath(
                outside,
                root: root)
        }
    }

    private func temporaryRoot(_ tag: String) -> String {
        GemmaFrozenPlan.temporaryRoot("range-plan-\(tag)")
    }

    /// This suite's own tests compare two plans against each other, so the
    /// shard identity only has to be self-consistent — the absolute path is
    /// fine here. Tests that pin a fingerprint must not do this; see
    /// `GemmaFrozenPlan`.
    private func parseHeader(path: String) throws -> Safetensors.Header {
        try GemmaFrozenPlan.parseHeader(realPath: path, shardID: path)
    }
}
