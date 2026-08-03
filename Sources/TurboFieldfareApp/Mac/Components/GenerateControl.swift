import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct GenerateControl: View {
    let model: AppModel
    private let controlHeight: CGFloat = 34

    var body: some View {
        if model.isRunningInActiveConversation {
            runningPill
        } else {
            // A run owned by another conversation shows the disabled Generate
            // button, never a Stop pill: the pill carries `.cancelAction`, so
            // putting it here would arm Escape against a generation the user
            // cannot even see.
            generateButton
        }
    }

    /// Nil unless a *different* conversation is generating.
    private var generatingElsewhereTitle: String? {
        guard let id = model.generatingConversationID,
              id != model.activeConversationID else { return nil }
        return model.conversations.first(where: { $0.id == id })?.title
    }

    private var generateHelp: String {
        guard let title = generatingElsewhereTitle else { return "Generate" }
        return "A reply is still generating in “\(title)”"
    }

    private var generateButton: some View {
        Button {
            model.run()
        } label: {
            Label("Generate", systemImage: "arrow.up")
                .font(.callout.weight(.semibold))
                .padding(.horizontal, 24)
                .frame(minWidth: 124, minHeight: controlHeight)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(TurboFieldfareMacTheme.accentColor, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!model.canRun)
        .opacity(model.canRun ? 1 : 0.62)
        .help(generateHelp)
    }

    private var runningPill: some View {
        Button {
            model.cancel()
        } label: {
            HStack(spacing: 10) {
                if model.isCancellationPending {
                    Text("Stopping")
                        .font(.callout.weight(.medium))
                } else if model.phase == .prefill {
                    Text(model.presentation.label)
                        .font(.callout.weight(.medium))
                        .monospacedDigit()
                        .contentTransition(.numericText())
                } else {
                    Text("\(MetricFormat.rate(model.liveTokensPerSecond)) tok/s")
                        .font(.callout.weight(.semibold))
                        .monospacedDigit()
                        .contentTransition(.numericText())
                }
                Label("Stop generation", systemImage: "stop.fill")
                    .labelStyle(.iconOnly)
                    .font(.callout)
                    .frame(width: 28, height: 28)
            }
            .padding(.leading, 18)
            .padding(.trailing, 4)
            .frame(minWidth: 140, minHeight: controlHeight)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(TurboFieldfareMacTheme.accentColor, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(.cancelAction)
        .disabled(!model.canCancel)
        .help("Stop generation")
        .animation(.smooth(duration: 0.2), value: model.presentation.label)
    }
}
