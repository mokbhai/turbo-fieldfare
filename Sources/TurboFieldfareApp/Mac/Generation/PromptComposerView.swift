import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if model.hasActiveSystemPrompt {
                systemPromptChip
            }
            editor
            footer
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 22)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    private var editor: some View {
        TextEditor(text: $model.promptText)
            .accessibilityLabel("Prompt")
            .font(.body)
            .scrollContentBackground(.hidden)
            .focused($promptFocused)
            .frame(height: editorHeight)
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    // Matches the NSTextView text origin: 5pt line fragment
                    // padding, no vertical inset.
                    Text("Prompt")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
    }

    private var editorHeight: CGFloat {
        model.promptText.isEmpty ? 46 : 84
    }

    /// Shows that this chat carries instructions the transcript does not, so a
    /// reply shaped by them is never a mystery.
    private var systemPromptChip: some View {
        Button(action: editSystemPrompt) {
            HStack(spacing: 6) {
                Image(systemName: "text.quote")
                    .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                Text(firstLine(of: model.activeSystemPrompt))
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .foregroundStyle(.secondary)
            }
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(TurboFieldfareMacTheme.accentColor.opacity(0.1), in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help("System prompt: \(model.activeSystemPrompt)")
        .accessibilityLabel("Edit system prompt")
    }

    private func firstLine(of text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: \.isNewline)
            .first.map(String.init) ?? ""
    }

    private func editSystemPrompt() {
        NotificationCenter.default.post(name: .turboFieldfareEditSystemPrompt, object: nil)
    }

    private var systemPromptButton: some View {
        Button(action: editSystemPrompt) {
            Label("System prompt", systemImage: model.hasActiveSystemPrompt
                  ? "text.bubble.fill" : "text.bubble")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(model.hasActiveSystemPrompt
                         ? TurboFieldfareMacTheme.accentColor : .secondary)
        .help(model.hasActiveSystemPrompt ? "Edit system prompt" : "Add a system prompt")
    }

    private var footer: some View {
        HStack(spacing: 10) {
            promptTips
            systemPromptButton
            Spacer()
            clearAction
            GenerateControl(model: model)
        }
    }

    private var promptTips: some View {
        Button {
            showingPromptTips.toggle()
        } label: {
            Label("Prompt tips", systemImage: "questionmark.circle")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help("Prompt tips")
        .popover(isPresented: $showingPromptTips,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            promptGuide
        }
    }

    private var promptGuide: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting this model")
                .font(.headline)

            tipSection("Ask for a clear task",
                       "Say what you want the model to create, explain, plan, or transform. Put the essential context in the same prompt.")
            tipSection("Shape the answer",
                       "Specify a useful length, sections, tone, or output format. Concrete constraints work better than a long list of vague preferences.")
            tipSection("Anchor important facts",
                       "Include facts the answer must preserve and say what should be checked. Generated factual claims can still be wrong or outdated.")
            tipSection("For code and calculations",
                       "Provide types, dimensions, interfaces, edge cases, or a small scaffold. Compile or run the result before relying on it.")
            tipSection("Try a focused revision",
                       "If the answer drifts, shorten the task and make the missing requirement explicit. The default temperature is 0.20 for steadier responses.")
        }
        .font(.callout)
        .frame(width: 390, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .fontWeight(.semibold)
            Text(detail)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var clearAction: some View {
        // Scoped to the conversation on screen: an idle chat must still be able
        // to clear its own transcript or composer while another one generates.
        if !model.isRunningInActiveConversation && model.hasOutputTranscript {
            Button {
                model.clearOutput()
            } label: {
                Label("Clear output", systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear output")
        } else if !model.isRunningInActiveConversation && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused = true
            } label: {
                Label("Clear prompt", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear prompt")
        }
    }
}
