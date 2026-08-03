import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// Tests `KVCacheManager` FP16 shape, growth, separate K/V storage, ring,
/// and reset semantics against the Gemma 4 config.
@Suite struct KVCacheManagerTests {

    private let config = ArchConfig.gemma4_26B_A4B

    private func makeManager(maxContext: Int,
                             fp16RingEnabled: Bool = false,
                             fp16RingCapacityOverride: Int? = nil) throws -> (MetalContext, KVCacheManager) {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: config,
                                    maxContext: maxContext,
                                    fp16RingEnabled: fp16RingEnabled,
                                    slidingWindow: config.slidingWindow,
                                    maxPrefillChunkTokens: 128,
                                    fp16RingCapacityOverride: fp16RingCapacityOverride)
        return (ctx, kv)
    }


    @Test func strideAndBufferSizes_matchConfig() throws {
        let (_, kv) = try makeManager(maxContext: 128)

        // SWA: numKVHeads(8) * headDim(256) * 2 = 4096 B/token.
        // Full: numFullKVHeads(2) * fullHeadDim(512) * 2 = 2048 B/token.
        #expect(kv.kRange(layer: 0, start: 0, count: 1).stride == 8 * 256 * 2)
        #expect(kv.kRange(layer: 5, start: 0, count: 1).stride == 2 * 512 * 2)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 128 * 4096)
        #expect(kv.keyBuffer(layer: 5, validTokenCount: 0).length == 128 * 2048)
    }

    @Test func linearGrowth_tracksAdvance() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.position == 0)
        for n in 1...100 {
            kv.advance()
            #expect(kv.position == n)
        }
    }

    /// Full layers share the raw k_proj output, then diverge: K runs k_norm +
    /// RoPE while V runs no-scale v_norm without RoPE. They therefore require
    /// separate cache slots.
    @Test func fullLayer_separatesKAndVBuffers() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        let k = kv.keyBuffer(layer: 5, validTokenCount: 0)
        let v = kv.valueBuffer(layer: 5, validTokenCount: 0)
        #expect(k !== v, "full-layer K and V must NOT alias")
        let ks = kv.kSlot(layer: 5, position: 3)
        let vs = kv.vSlot(layer: 5, position: 3)
        #expect(ks.buffer !== vs.buffer, "full-layer K/V slots must NOT alias")
        // Offsets are still per-position-strided in both buffers.
        #expect(ks.offset == vs.offset)
    }

    @Test func swaLayer_hasSeparateKVBuffers() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0)
                !== kv.valueBuffer(layer: 0, validTokenCount: 0))
    }

    @Test func slotOffsets_areLinear() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.kSlot(layer: 0, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 0, position: 3).offset == 3 * 4096)
        #expect(kv.vSlot(layer: 5, position: 7).offset == 7 * 2048)
    }

    @Test func fp16Ring_capsSWALayersAndLeavesFullLayersLinear() throws {
        let (_, kv) = try makeManager(maxContext: 4096,
                                      fp16RingEnabled: true)

        #expect(kv.fp16RingEnabled)
        #expect(kv.capacity(layer: 0) == 1152)
        #expect(kv.ringCapacity(layer: 0) == 1152)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 1152 * 4096)
        #expect(kv.capacity(layer: 5) == 4096)
        #expect(kv.ringCapacity(layer: 5) == 0)
        #expect(kv.keyBuffer(layer: 5, validTokenCount: 0).length == 4096 * 2048)
    }

    @Test func fp16Ring_shortSessionCapsSWAToMaxContext() throws {
        let (_, kv) = try makeManager(maxContext: 256,
                                      fp16RingEnabled: true)

        #expect(kv.fp16RingEnabled)
        #expect(kv.capacity(layer: 0) == 256)
        #expect(kv.ringCapacity(layer: 0) == 256)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 256 * 4096)
        #expect(kv.capacity(layer: 5) == 256)
        #expect(kv.ringCapacity(layer: 5) == 0)
        #expect(kv.keyBuffer(layer: 5, validTokenCount: 0).length == 256 * 2048)
    }

    @Test func fp16Ring_slotOffsetsWrapOnlyForSWALayers() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      fp16RingEnabled: true,
                                      fp16RingCapacityOverride: 32)

        #expect(kv.kSlot(layer: 0, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 0, position: 31).offset == 31 * 4096)
        #expect(kv.kSlot(layer: 0, position: 32).offset == 0)
        #expect(kv.vSlot(layer: 0, position: 35).offset == 3 * 4096)

        #expect(kv.kSlot(layer: 5, position: 35).offset == 35 * 2048)
        #expect(kv.vSlot(layer: 5, position: 35).offset == 35 * 2048)
    }

    @Test func fp16Ring_rangesMustNotWrap() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      fp16RingEnabled: true,
                                      fp16RingCapacityOverride: 32)

        let k = kv.kRange(layer: 0, start: 28, count: 4)
        #expect(k.offset == 28 * 4096)
        let v = kv.vRange(layer: 0, start: 32, count: 3)
        #expect(v.offset == 0)
    }

    @Test func rangeSlotsHaveLinearOffsets() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        let swaStride = kv.kRange(layer: 0, start: 0, count: 1).stride
        let fullStride = kv.vRange(layer: 5, start: 0, count: 1).stride

        let k = kv.kRange(layer: 0, start: 7, count: 3)
        let v = kv.vRange(layer: 5, start: 11, count: 5)

        #expect(k.offset == 7 * swaStride)
        #expect(k.stride == swaStride)
        #expect(v.offset == 11 * fullStride)
        #expect(v.stride == fullStride)
        #expect(k.buffer === kv.keyBuffer(layer: 0, validTokenCount: 0))
        #expect(v.buffer === kv.valueBuffer(layer: 5, validTokenCount: 0))
    }

    @Test func advanceByCountTracksCursor() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 31)
        #expect(kv.position == 31)
        kv.advance(by: 0)
        #expect(kv.position == 31)
        kv.advance()
        #expect(kv.position == 32)
    }

    @Test func reset_clearsPosition() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        for _ in 0..<100 { kv.advance() }
        #expect(kv.position == 100)
        kv.reset()
        #expect(kv.position == 0)
        // Cursor reusable after reset.
        kv.advance()
        #expect(kv.position == 1)
    }

    /// The bound is a rewind *distance*, not an absolute cursor: a ring holding
    /// its window `W` in capacity `C` has `C - W` tokens of slack, and it has
    /// that slack however far past `C` the cursor has run.
    @Test func rewindOfExactlyTheRingSlackIsAllowedLongAfterWrapping() throws {
        let (_, kv) = try makeManager(maxContext: 4096,
                                      fp16RingEnabled: true)
        let slack = kv.capacity(layer: 0) - config.slidingWindow  // 1152 - 1024
        #expect(slack == 128)
        // Well past the ring capacity — the old absolute bound refused this
        // outright, which is exactly where prompt reuse matters most.
        kv.advance(by: 3000)

        #expect(kv.canRewind(to: 3000 - slack))
        kv.rewind(to: 3000 - slack)
        #expect(kv.position == 3000 - slack)
        kv.advance(by: slack)
        #expect(kv.position == 3000)
    }

    /// One token past `C - W` is refused. Residency would in fact still hold
    /// there — the true maximum is `C - W + 1` — so this pins the margin the
    /// bound keeps on purpose, not the point where the ring runs out.
    @Test func rewindOneTokenPastTheRingSlackIsRefused() throws {
        let (_, kv) = try makeManager(maxContext: 4096,
                                      fp16RingEnabled: true)
        let slack = kv.capacity(layer: 0) - config.slidingWindow
        kv.advance(by: 3000)

        #expect(!kv.canRewind(to: 3000 - slack - 1))
    }

    /// A ring too small to hold its own window has no slack at all, so every
    /// rewind that actually *moves* the cursor fails closed rather than silently
    /// attending over evicted slots. The no-op rewind stays legal.
    @Test func rewindIsRefusedWhenTheRingCannotHoldItsWindow() throws {
        let (_, kv) = try makeManager(maxContext: 4096,
                                      fp16RingEnabled: true,
                                      fp16RingCapacityOverride: 8)
        #expect(kv.capacity(layer: 0) - config.slidingWindow < 0)
        kv.advance(by: 20)

        #expect(!kv.canRewind(to: 19))
        #expect(!kv.canRewind(to: 0))
        // But the zero-distance rewind stays legal even here. It moves nothing,
        // so nothing can be evicted by it; refusing it would make `rewind` trip
        // its own precondition and abort the process on a plain no-op.
        #expect(kv.canRewind(to: 20))
        kv.rewind(to: 20)
        #expect(kv.position == 20)
    }

    /// Ring off, every layer sized for the whole context: nothing can wrap, so
    /// no distance limit applies.
    @Test func rewindIsUnboundedWhenNoLayerCanWrap() throws {
        let (_, kv) = try makeManager(maxContext: 256)
        kv.advance(by: 200)

        #expect(kv.canRewind(to: 0))
        kv.rewind(to: 1)
        #expect(kv.position == 1)
    }

    /// The arithmetic tests above would pass for any bound of the same shape.
    /// This one reads the ring itself: it stamps every position with its own
    /// index, rewinds by the permitted slack, and checks that each slot the
    /// re-prefill's attention window would touch still answers with the logical
    /// position it claims. A bound one token too loose evicts the oldest such
    /// slot, and the marker there comes back as a *different* position — which
    /// is exactly the corruption `canRewind` exists to prevent.
    ///
    /// No model and no kernels: markers stand in for K vectors, written through
    /// the manager's own `kSlot`, so the ring's real address arithmetic is under
    /// test.
    @Test func rewindOfThePermittedSlackLeavesTheWholeAttentionWindowResident() throws {
        let (_, kv) = try makeManager(maxContext: 4096, fp16RingEnabled: true)
        let capacity = kv.capacity(layer: 0)          // 1152, the SWA ring
        let window = config.slidingWindow             // 1024
        let cursor = 3000                             // well past one wrap
        for position in 0..<cursor {
            writeMarker(kv, layer: 0, position: position)
        }
        kv.advance(by: cursor)

        // Rewind as far as the manager itself will go, rather than by a
        // hardcoded slack: probing is what makes the residency check below bite
        // on a bound that is too loose instead of quietly agreeing with it.
        var permitted = 0
        while kv.canRewind(to: cursor - (permitted + 1)) { permitted += 1 }
        // `C - W`, one token inside the true maximum `C - W + 1`. That last
        // token is refused on purpose, not because residency fails there.
        #expect(permitted == capacity - window)       // 128

        kv.rewind(to: cursor - permitted)

        // Re-prefilling queries `[position, cursor)` reads keys back to
        // `position - W + 1`; everything from there up must survive.
        let oldestKeyRead = kv.position - window + 1
        var stale: [Int] = []
        for position in oldestKeyRead..<cursor
        where readMarker(kv, layer: 0, position: position) != Int32(position) {
            stale.append(position)
        }
        #expect(stale.isEmpty, "evicted slots inside the re-prefill window: \(stale.prefix(8))")

        // And the test can tell a wrong bound from a right one: two tokens past
        // the permitted slack the window reaches a slot the ring really has
        // recycled, and it answers with the newer position that overwrote it.
        let evicted = oldestKeyRead - 2
        #expect(readMarker(kv, layer: 0, position: evicted) == Int32(evicted + capacity))
    }

    /// Stamp a slot with its own logical position. Only the first four bytes of
    /// the token's stride are used; residency is about *which* position owns the
    /// slot, not about the vector's contents.
    private func writeMarker(_ kv: KVCacheManager, layer: Int, position: Int) {
        let slot = kv.kSlot(layer: layer, position: position)
        slot.buffer.contents().advanced(by: slot.offset)
            .assumingMemoryBound(to: Int32.self).pointee = Int32(position)
    }

    private func readMarker(_ kv: KVCacheManager, layer: Int, position: Int) -> Int32 {
        let slot = kv.kSlot(layer: layer, position: position)
        return slot.buffer.contents().advanced(by: slot.offset)
            .assumingMemoryBound(to: Int32.self).pointee
    }

}
