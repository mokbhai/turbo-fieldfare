import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct PackedExpertsLayoutTests {

    /// Hand-write a tiny layout.json with one layer, two experts, two
    /// sub-tensors each. Returns the directory URL.
    ///
    /// `overrides` replaces top-level keys, which is how the bounds tests state
    /// a corrupt `expertsPerLayer` without a second fixture.
    static func writeToyLayout(_ overrides: [String: Any] = [:]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-layout-test-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        var root: [String: Any] = [
            "expertStride": 16384,
            "numLayers": 1,
            "expertsPerLayer": 2,
            "layers": [
                [
                    "layer": 0,
                    "file": "layer_00.bin",
                    "experts": [
                        [
                            "expert": 0,
                            "offset": 0,
                            "size": 16384,
                            "tensors": [
                                "gate": [
                                    "offset": 0,
                                    "size": 4096,
                                    "dtype": "U32",
                                    "shape": [64, 64],
                                    "bits": 4
                                ],
                                "gate_scales": [
                                    "offset": 4096,
                                    "size": 256,
                                    "dtype": "BF16",
                                    "shape": [64, 1]
                                ],
                            ],
                        ],
                        [
                            "expert": 1,
                            "physicalRank": 1,
                            "offset": 16384,
                            "size": 16384,
                            "tensors": [
                                "gate": [
                                    "offset": 0,
                                    "size": 4096,
                                    "dtype": "U32",
                                    "shape": [64, 64],
                                    "bits": 4
                                ],
                                "gate_scales": [
                                    "offset": 4096,
                                    "size": 256,
                                    "dtype": "BF16",
                                    "shape": [64, 1]
                                ],
                            ],
                        ],
                    ],
                ],
            ],
        ]
        for (key, value) in overrides { root[key] = value }
        let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        try data.write(to: exp.appendingPathComponent("layout.json"))
        return dir
    }

    @Test func decodesToyLayout() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        let layout = try PackedExpertsLayoutReader.load(directoryURL: dir)
        #expect(layout.expertStride == 16384)
        #expect(layout.numLayers == 1)
        #expect(layout.expertsPerLayer == 2)
        #expect(layout.layers.count == 1)
        let exp0 = layout.expert(layer: 0, expert: 0)
        #expect(exp0.offset == 0)
        #expect(exp0.size == 16384)
        let gate = try #require(exp0.subTensors["gate"])
        #expect(gate.offset == 0)
        #expect(exp0.subTensors["gate_scales"]?.offset == 4096)
        let exp1 = layout.expert(layer: 0, expert: 1)
        #expect(exp1.offset == 16384)
        #expect(exp1.expert == 1)
    }

    @Test func missingLayoutJsonThrowsMissingFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-no-layout-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try PackedExpertsLayoutReader.load(directoryURL: dir)
        } throws: { error in
            if case ModelError.missingFile = error { return true }
            return false
        }
    }

    /// **`expertsPerLayer` is an allocation count read out of a file.**
    ///
    /// `[ExpertEntry?](repeating: nil, count: expertsPerLayer)` TRAPS on a
    /// negative count — the process aborts, taking the app with it, and nothing
    /// reports which file was corrupt. Same class as the manifest traps already
    /// fixed (`manifest.numLayers: -1`, `fullAttentionLayerMask: [-1]`), and
    /// reached the same way: `layout.json` is a file on disk that the loader
    /// reads before anything has compared it to the manifest.
    ///
    /// `maxExpertsPerLayer + 1` is in the table for the other half: it does not
    /// trap, it allocates, and a 16 MiB layout can ask for an arbitrarily large
    /// one.
    @Test(arguments: [-1, PackedExpertsLayoutReader.maxExpertsPerLayer + 1])
    func outOfRangeExpertsPerLayerIsReportedRatherThanTrapping(_ experts: Int) throws {
        let dir = try Self.writeToyLayout(["expertsPerLayer": experts])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try PackedExpertsLayoutReader.load(directoryURL: dir)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("expertsPerLayer is out of range")
        }
    }

    /// Zero is a real answer — a synthetic snapshot with no routed experts
    /// writes it, and `GTurboJSON.encodeLayout` emits `?? 0` when there are no
    /// layer plans. Pinned so the bound above is a bound and not an accidental
    /// `> 0`.
    @Test func zeroExpertsPerLayerStillLoads() throws {
        let dir = try Self.writeToyLayout([
            "expertsPerLayer": 0,
            "layers": [["layer": 0, "file": "layer_00.bin", "experts": [] as [Any]]],
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let layout = try PackedExpertsLayoutReader.load(directoryURL: dir)
        #expect(layout.expertsPerLayer == 0)
        #expect(layout.layers.first?.experts.isEmpty == true)
    }

    @Test func negativeLayerCountIsReported() throws {
        let dir = try Self.writeToyLayout(["numLayers": -1])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try PackedExpertsLayoutReader.load(directoryURL: dir)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("numLayers is negative")
        }
    }

    @Test func oversizedLayoutRejectsBeforeDecode() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        let layoutURL = dir
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layout.json")
        try Data(repeating: 0x20, count: 64).write(to: layoutURL)

        #expect {
            _ = try PackedExpertsLayoutReader.load(directoryURL: dir,
                                                   maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }
}
