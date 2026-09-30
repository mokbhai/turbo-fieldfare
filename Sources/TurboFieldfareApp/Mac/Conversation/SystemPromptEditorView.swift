import TurboFieldfareAppCore
import SwiftUI

/// Edits the system prompt of the conversation on screen. Works on a copy and
/// writes it back only on Save, so the store is not rewritten per keystroke.
struct SystemPromptEditorView: View {
    let model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var text: String
    @State private var alsoMakeDefault = false
    @FocusState private var editorFocused: Bool

    init(model: AppModel) {
        self.model = model
        _text = State(initialValue: model.activeSystemPrompt)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("System Prompt")
                    .font(.title3.weight(.semibold))
                Text("Instructions the model follows for every reply in this chat, such as a role, tone, or output format.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            TextEditor(text: $text)
                .font(.body)
                .focused($editorFocused)
                .scrollContentBackground(.hidden)
                .padding(8)
                .frame(minHeight: 180)
                .background {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color(nsColor: .textBackgroundColor))
                        .overlay {
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(.separator, lineWidth: 0.5)
                        }
                }
                .overlay(alignment: .topLeading) {
                    if text.isEmpty {
                        Text("You are a concise assistant. Answer in plain English and use Markdown lists for steps.")
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 13)
                            .padding(.vertical, 8)
                            .allowsHitTesting(false)
                    }
                }

            HStack(spacing: 8) {
                Menu("Templates") {
                    ForEach(SystemPromptTemplate.all) { template in
                        Button(template.title) { text = template.prompt }
                    }
                }
                .fixedSize()
                if !model.defaultSystemPrompt.isEmpty && text != model.defaultSystemPrompt {
                    Button("Use Default") { text = model.defaultSystemPrompt }
                }
                Spacer()
                Text("≈\(estimatedTokens) tokens")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Toggle("Also use for new chats", isOn: $alsoMakeDefault)
                .toggleStyle(.checkbox)

            Text("Changing the system prompt re-reads the whole conversation on the next reply, so that reply starts more slowly.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack {
                Button("Clear", role: .destructive) { text = "" }
                    .disabled(text.isEmpty)
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") { save() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 520)
        .onAppear { editorFocused = true }
    }

    private var estimatedTokens: Int {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? 0 : Int((Double(trimmed.count) / 3.5).rounded(.up))
    }

    private func save() {
        model.setActiveSystemPrompt(text)
        if alsoMakeDefault {
            model.setDefaultSystemPrompt(text)
        }
        dismiss()
    }
}

private struct SystemPromptTemplate: Identifiable {
    let id: String
    let title: String
    let prompt: String

    static let all: [SystemPromptTemplate] = [
        SystemPromptTemplate(
            id: "concise",
            title: "Concise assistant",
            prompt: "You are a helpful assistant. Answer directly and concisely. Use short paragraphs, and use Markdown lists only when they make steps or options clearer."),
        SystemPromptTemplate(
            id: "coder",
            title: "Programming partner",
            prompt: "You are an experienced software engineer. Give correct, idiomatic code in fenced code blocks with the language named. Explain trade-offs briefly and point out edge cases the user should test."),
        SystemPromptTemplate(
            id: "tutor",
            title: "Patient tutor",
            prompt: "You are a patient tutor. Explain ideas step by step, starting from what the learner already knows. Use a small example, then check understanding with one short question."),
        SystemPromptTemplate(
            id: "editor",
            title: "Writing editor",
            prompt: "You are a careful editor. Improve clarity, grammar, and flow while keeping the author's voice and meaning. Return the revised text first, then a short list of the main changes."),
    ]
}
