import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppDiagnosticsTests {
    @Test func requestStartTTFTAddsPrefillAndPostPrefillWait() {
        let diagnostics = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            prefillSeconds: 1.25,
            timeToFirstTokenSeconds: 0.5,
            decodeSeconds: 0.75,
            tokensPerSecond: 1.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())

        #expect(diagnostics.requestStartTimeToFirstTokenSeconds == 1.75)
    }

    @Test func requestStartTTFTIsNilWhenEitherSideIsMissing() {
        let missingPrefill = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            prefillSeconds: nil,
            timeToFirstTokenSeconds: 0.5,
            decodeSeconds: 0.75,
            tokensPerSecond: 1.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        let missingFirstToken = AppDiagnostics(
            generatedTokens: 0,
            stopReason: .cancelled,
            prefillSeconds: 1.25,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())

        #expect(missingPrefill.requestStartTimeToFirstTokenSeconds == nil)
        #expect(missingFirstToken.requestStartTimeToFirstTokenSeconds == nil)
    }

    @Test func promptPrefillRateDividesPromptTokensByPrefillSeconds() throws {
        let diagnostics = AppDiagnostics(
            generatedTokens: 8,
            stopReason: .eos,
            promptTokenCount: 20,
            prefillSeconds: 6.17,
            timeToFirstTokenSeconds: 0.1,
            decodeSeconds: 1.0,
            tokensPerSecond: 8.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())

        let rate = try #require(diagnostics.promptPrefillTokensPerSecond)
        // Pins the seconds unit: milliseconds would give ~0.0032 instead.
        #expect(abs(rate - 3.2415) < 0.001)
    }

    @Test func promptPrefillRateIsNilWhenInputsAreUnusable() {
        let missingPromptTokens = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            promptTokenCount: nil,
            prefillSeconds: 6.17,
            timeToFirstTokenSeconds: 0.1,
            decodeSeconds: 1.0,
            tokensPerSecond: 8.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        let missingPrefill = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            promptTokenCount: 20,
            prefillSeconds: nil,
            timeToFirstTokenSeconds: 0.1,
            decodeSeconds: 1.0,
            tokensPerSecond: 8.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        // A fully cache-resumed prefill can floor to zero; dividing would be inf.
        let zeroPrefill = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            promptTokenCount: 20,
            prefillSeconds: 0,
            timeToFirstTokenSeconds: 0.1,
            decodeSeconds: 1.0,
            tokensPerSecond: 8.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())

        // Wall-clock timing means a negative interval is representable; a
        // negative rate would render as a plausible-looking "-3.2 tok/s".
        let negativePrefill = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            promptTokenCount: 20,
            prefillSeconds: -6.17,
            timeToFirstTokenSeconds: 0.1,
            decodeSeconds: 1.0,
            tokensPerSecond: 8.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())

        #expect(missingPromptTokens.promptPrefillTokensPerSecond == nil)
        #expect(missingPrefill.promptPrefillTokensPerSecond == nil)
        #expect(zeroPrefill.promptPrefillTokensPerSecond == nil)
        #expect(negativePrefill.promptPrefillTokensPerSecond == nil)
    }

    @Test func runnerDiagnosticsRetainPublicResultAndAdvancedMetrics() {
        let diagnostics = AppRunnerDiagnostics(
            cb1MillisecondsPerToken: 1,
            ioMillisecondsPerToken: 2,
            cb2MillisecondsPerToken: 3,
            headMillisecondsPerToken: 4,
            rdadviseMillisecondsPerToken: 5,
            rdadviseCallsPerToken: 6,
            rdadviseMegabytesPerToken: 7,
            rdadviseSkippedPerToken: 8,
            rdadviseFailures: 9)

        #expect(diagnostics.cb1MillisecondsPerToken == 1)
        #expect(diagnostics.ioMillisecondsPerToken == 2)
        #expect(diagnostics.cb2MillisecondsPerToken == 3)
        #expect(diagnostics.headMillisecondsPerToken == 4)
        #expect(diagnostics.rdadviseMillisecondsPerToken == 5)
        #expect(diagnostics.rdadviseCallsPerToken == 6)
        #expect(diagnostics.rdadviseMegabytesPerToken == 7)
        #expect(diagnostics.rdadviseSkippedPerToken == 8)
        #expect(diagnostics.rdadviseFailures == 9)
    }
}
