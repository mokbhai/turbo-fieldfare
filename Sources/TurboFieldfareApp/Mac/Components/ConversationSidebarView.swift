import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ConversationSidebarView: View {
    let model: AppModel
    @State private var renamingID: UUID?
    @State private var renameText = ""
    @State private var pendingDeletionID: UUID?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            list
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text("Chats")
                .font(.headline)
            Spacer()
            Button(action: model.newConversation) {
                Label("New Chat", systemImage: "square.and.pencil")
                    .labelStyle(.iconOnly)
                    .frame(width: 26, height: 26)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.borderless)
            .help("New Chat")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                ForEach(model.conversationsByRecency) { conversation in
                    row(conversation)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
        }
    }

    @ViewBuilder
    private func row(_ conversation: Conversation) -> some View {
        let isActive = conversation.id == model.activeConversationID
        Group {
            if renamingID == conversation.id {
                TextField("Title", text: $renameText)
                    .textFieldStyle(.roundedBorder)
                    .font(.callout)
                    .onSubmit { commitRename(conversation.id) }
                    .onExitCommand { renamingID = nil }
            } else {
                Button {
                    model.selectConversation(conversation.id)
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(conversation.title)
                            .font(.callout)
                            .lineLimit(1)
                            .truncationMode(.tail)
                            .foregroundStyle(isActive ? Color.primary : .secondary)
                        Text(subtitle(conversation))
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .background {
                        RoundedRectangle(cornerRadius: 7)
                            .fill(isActive
                                  ? TurboFieldfareMacTheme.accentColor.opacity(0.14)
                                  : Color.clear)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .contextMenu {
                    Button("Rename…") {
                        renameText = conversation.title
                        renamingID = conversation.id
                    }
                    Button("Delete", role: .destructive) {
                        pendingDeletionID = conversation.id
                    }
                }
            }
        }
        .confirmationDialog(
            "Delete this conversation?",
            isPresented: Binding(
                get: { pendingDeletionID == conversation.id },
                set: { if !$0 { pendingDeletionID = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                model.deleteConversation(conversation.id)
                pendingDeletionID = nil
            }
            Button("Cancel", role: .cancel) { pendingDeletionID = nil }
        } message: {
            Text(deletionMessage(conversation))
        }
    }

    /// Deleting a generating chat stops its reply, so the confirmation has to
    /// say so: the run is not visible from every other chat, and losing it is
    /// the part of the deletion the user cannot undo by reading history.
    private func deletionMessage(_ conversation: Conversation) -> String {
        let removal = "\(conversation.title) will be removed permanently."
        guard model.isGenerating(conversation.id) else { return removal }
        return removal + " The reply still generating in it will stop."
    }

    private func subtitle(_ conversation: Conversation) -> String {
        // Marks the generating row in text rather than with a pulsing dot: same
        // information, without an animation running inside a LazyVStack for the
        // length of a 26B generation.
        let generating = model.isGenerating(conversation.id)
        guard !conversation.isEmpty else { return generating ? "Generating…" : "Empty" }
        let exchanges = conversation.turns.filter { $0.role == .user }.count
        let count = exchanges == 1 ? "1 message" : "\(exchanges) messages"
        // Keep the count while generating: it is what the user scans the list
        // by, and losing it for the length of a 26B run makes a busy chat
        // harder to find than an idle one.
        return generating ? "\(count) · generating…" : count
    }

    private func commitRename(_ id: UUID) {
        model.renameConversation(id, to: renameText)
        renamingID = nil
    }
}
