import AppKit
import TurboFieldfareAppCore
import UniformTypeIdentifiers

/// Saves the conversation on screen as a Markdown file the user picks.
@MainActor
enum ConversationExport {
    static func exportActiveConversation(of model: AppModel) {
        guard let markdown = model.activeConversationMarkdown else { return }
        let panel = NSSavePanel()
        panel.title = "Export Conversation"
        panel.allowedContentTypes = [UTType(filenameExtension: "md") ?? .plainText]
        panel.nameFieldStringValue = fileName(for: model.activeConversation?.title)
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try Data(markdown.utf8).write(to: url, options: .atomic)
        } catch {
            NSAlert(error: error).runModal()
        }
    }

    private static func fileName(for title: String?) -> String {
        let base = (title ?? Conversation.untitled)
            .components(separatedBy: CharacterSet(charactersIn: "/:\\?%*|\"<>"))
            .joined(separator: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (base.isEmpty ? Conversation.untitled : base) + ".md"
    }
}

extension Notification.Name {
    /// Posted by the menu bar, the composer, and the inspector; the sheet lives
    /// in the root view, which owns the window.
    static let turboFieldfareEditSystemPrompt = Notification.Name(
        "com.turbofieldfare.editSystemPrompt")
}
