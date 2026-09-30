import Foundation

public struct StreamingStopMatcher: Sendable {
    private let stops: [String]
    private var pending = ""
    public private(set) var isStopped = false
    /// The stop string that ended the stream, when one did. Earliest match
    /// wins; among matches at the same position, the first listed.
    public private(set) var matchedStop: String?

    public init(stops: [String]) {
        self.stops = stops.filter { !$0.isEmpty }
    }

    public mutating func push(_ text: String) -> String {
        guard !isStopped else { return "" }
        pending += text
        if let match = earliestMatch(in: pending) {
            let output = String(pending[..<match.index])
            pending = ""
            isStopped = true
            matchedStop = match.stop
            return output
        }
        let retained = longestPossibleSuffix(in: pending)
        let boundary = pending.index(pending.endIndex, offsetBy: -retained)
        let output = String(pending[..<boundary])
        pending = String(pending[boundary...])
        return output
    }

    public mutating func finish() -> String {
        guard !isStopped else { return "" }
        defer { pending = "" }
        return pending
    }

    private func earliestMatch(in text: String) -> (index: String.Index, stop: String)? {
        var best: (index: String.Index, stop: String)?
        for stop in stops {
            guard let index = text.range(of: stop)?.lowerBound else { continue }
            if best == nil || index < best!.index { best = (index, stop) }
        }
        return best
    }

    private func longestPossibleSuffix(in text: String) -> Int {
        var best = 0
        for stop in stops {
            let maximum = min(text.count, max(stop.count - 1, 0))
            for length in stride(from: maximum, through: 1, by: -1) {
                if text.suffix(length) == stop.prefix(length) {
                    best = max(best, length)
                    break
                }
            }
        }
        return best
    }
}
