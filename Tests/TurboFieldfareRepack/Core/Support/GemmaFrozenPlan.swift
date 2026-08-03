import Darwin
import Foundation
@testable import TurboFieldfareRepackCore

/// Builds the one Gemma plan the M0 byte-identity tripwires are pinned against.
///
/// Every knob here feeds `RangeCopyPlan.canonicalFingerprint`, so this builder
/// exists to make sure the golden tests and any future caller agree on all of
/// them. Two are load-bearing and easy to get wrong:
///
/// * `shardID` is the **bare shard filename**, not the absolute snapshot path.
///   `RangeCopyPlanner` uses `SourceTensor.shardPath` as the shard identity and
///   hashes it into the fingerprint, and production only ever supplies the file
///   name (`RemoteSnapshotLoader` parses headers with the shard's name). Feeding
///   an absolute temporary path would embed a per-run UUID and the golden would
///   differ on every run.
/// * `rangeChunkBytes` is hashed too, which is why the goldens pin two of them
///   (`smallChunkBytes` and `productionChunkBytes`) rather than one. See the
///   note on those properties.
///
/// `SyntheticSnapshot`'s default seed is used because the fingerprint covers
/// names, offsets and sizes only — never payload bytes — so it is seed
/// independent. The seed still fixes the bytes, which keeps everything else
/// about the snapshot reproducible.
enum GemmaFrozenPlan {

    /// Shard identity as production supplies it: the file name alone.
    static let shardID = "model-00001-of-00001.safetensors"

    /// Small enough that the plan really exercises splitting and coalescing:
    /// `SyntheticSnapshot` downloads ~300 KB in total, so 4096 forces
    /// `splitLargeCopies` to cut tensors up and `coalesce` to stitch adjacent
    /// pieces back together. Nothing production ever uses — see
    /// `productionChunkBytes`.
    static let smallChunkBytes = 4096

    /// The chunk size production actually installs with. Hashed into
    /// `canonicalFingerprint` like any other, so pinning a fingerprint at
    /// `smallChunkBytes` alone leaves `RemoteChunkPolicy.defaultBytes` free to
    /// move: halving it to 32 MiB rewrites every production fingerprint — and
    /// so discards every in-flight resume checkpoint — without reddening a
    /// single assertion.
    ///
    /// Neither pin subsumes the other, which is why both goldens exist. At
    /// 64 MiB this snapshot fits in one range, so the split/coalesce ordering
    /// the small pin covers collapses away here; at 4096 the production
    /// constant is not in the hash at all. Together they cover the layout and
    /// the policy.
    static let productionChunkBytes = RemoteChunkPolicy.defaultBytes

    /// Build the repack plan and its range-copy plan, then tear the temporary
    /// directories down. Both plans are value types, so the closure result may
    /// safely outlive the snapshot on disk.
    static func withPlans<T>(
        rangeChunkBytes: Int = smallChunkBytes,
        _ body: (RepackPlan, RangeCopyPlan) throws -> T
    ) throws -> T {
        let snapshotDirectory = temporaryRoot("m0-snapshot")
        let outputDirectory = temporaryRoot("m0-output")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDirectory)
            try? FileManager.default.removeItem(atPath: outputDirectory)
        }
        let snapshot = try SyntheticSnapshot.build(at: snapshotDirectory)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDirectory)
        let arch = try ArchInfo.load(
            configPath: (snapshotDirectory as NSString)
                .appendingPathComponent("config.json"))
        let header = try parseHeader(realPath: snapshot.shardPath, shardID: shardID)
        let repackPlan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: outputDirectory)
        let rangePlan = try RangeCopyPlanner.plan(
            repackPlan: repackPlan,
            rangeChunkBytes: rangeChunkBytes)
        return try body(repackPlan, rangePlan)
    }

    static func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    /// Read the safetensors header from `realPath` but label every tensor in it
    /// with `shardID`. The two differ whenever a test wants a stable shard
    /// identity out of a temporary file — see the note above.
    static func parseHeader(realPath: String, shardID: String) throws
        -> Safetensors.Header {
        let fd = try Posix.openRead(realPath)
        defer { close(fd) }
        var headerSize: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSize) {
            try Posix.preadAll(
                fd: fd,
                path: realPath,
                buf: $0.baseAddress!,
                count: 8,
                offset: 0)
        }
        headerSize = UInt64(littleEndian: headerSize)
        var headerData = Data(count: Int(headerSize))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(
                fd: fd,
                path: realPath,
                buf: $0.baseAddress!,
                count: $0.count,
                offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: shardID,
            fileSize: try Posix.fileSize(fd: fd, path: realPath),
            headerBytes: headerData)
    }
}
