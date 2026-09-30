import AppKit
import TurboFieldfareAppCore
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
            Section {
                ModelPickerView(model: model)
            }
            modelSection
            memorySection
            systemPromptSection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var modelSection: some View {
        Section("Model") {
            LabeledContent("Path") {
                HStack(spacing: 6) {
                    Text(model.modelPathText)
                        .font(.caption)
                        .truncationMode(.middle)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                        .help(model.modelPathText)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.modelPathText, forType: .string)
                    } label: {
                        Label("Copy model path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy model path")
                }
            }
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
            }
            LabeledContent("State") {
                Text(model.presentation.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.requiresModelInstallation {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(model.installDescriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Installed size") {
                    Text(MetricFormat.storage(model.installDescriptor.installedBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let requirement = model.installRequirement {
                    LabeledContent("Available") {
                        Text(MetricFormat.storage(requirement.availableBytes))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    private var contextOptions: [AppContextOption] {
        AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: AppMemoryBudget.residentWeightEstimateBytes,
            installedRAMBytes: AppMemoryBudget.installedRAMBytes)
    }

    private var memorySection: some View {
        Section("Memory") {
            LabeledContent("Context") {
                Picker("Context", selection: $model.maxContextTokens) {
                    // Lengths the budget cannot hold stay listed but disabled,
                    // so the user can see what more RAM would buy.
                    ForEach(contextOptions) { option in
                        Text(option.label)
                            .help(option.disabledReason ?? "")
                            .tag(option.tokens)
                            .disabled(!option.isEnabled)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            ContextMeterView(model: model)
            LabeledContent("Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            Text("More slots can improve decode speed by keeping more experts in memory, but they also use more RAM. Changes are compared with 4K context and 16 slots and apply after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var systemPromptSection: some View {
        Section("System Prompt") {
            if model.hasActiveSystemPrompt {
                Text(model.activeSystemPrompt)
                    .font(.callout)
                    .lineLimit(4)
                    .truncationMode(.tail)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            } else {
                Text("None for this chat.")
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            }
            HStack {
                Button(model.hasActiveSystemPrompt ? "Edit…" : "Add…") {
                    NotificationCenter.default.post(
                        name: .turboFieldfareEditSystemPrompt, object: nil)
                }
                Spacer()
                if model.hasActiveSystemPrompt {
                    Button("Remove") { model.setActiveSystemPrompt("") }
                }
            }
            if !model.defaultSystemPrompt.isEmpty {
                LabeledContent("New chats") {
                    Button("Stop using default") { model.setDefaultSystemPrompt("") }
                        .buttonStyle(.link)
                        .font(.caption)
                }
                .help(model.defaultSystemPrompt)
            }
        }
    }

    private var presetBinding: Binding<AppSamplingPreset?> {
        Binding(
            get: { model.activeSamplingPreset },
            set: { preset in
                if let preset { model.applySamplingPreset(preset) }
            })
    }

    private var generationSection: some View {
        Section {
            LabeledContent("Preset") {
                Picker("Preset", selection: presetBinding) {
                    ForEach(AppSamplingPreset.allCases) { preset in
                        Text(preset.label)
                            .help(preset.detail)
                            .tag(Optional(preset))
                    }
                    if model.activeSamplingPreset == nil {
                        Text("Custom").tag(AppSamplingPreset?.none)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                    .fixedSize()
                }
            }
            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .disabled(!model.topKEnabled)
            if model.topKEnabled && model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }
            LabeledContent("Repetition penalty") {
                HStack(spacing: 8) {
                    Slider(value: $model.repetitionPenalty, in: 1...2, step: 0.05)
                    Text(model.repetitionPenalty, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Text("Above 1.00 discourages repeating earlier tokens. Keep it low: large values distort wording and code.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Limit response length", isOn: $model.maxResponseTokensEnabled)
                .toggleStyle(.switch)
            if model.maxResponseTokensEnabled {
                LabeledContent("Max tokens") {
                    HStack(spacing: 6) {
                        TextField("Max tokens", value: maxResponseTokensBinding,
                                  format: .number.grouping(.never))
                            .multilineTextAlignment(.trailing)
                            .monospacedDigit()
                            .frame(width: 64)
                            .labelsHidden()
                        Stepper("Max tokens", value: maxResponseTokensBinding,
                                in: 16...model.maxContextTokens, step: 256)
                            .labelsHidden()
                    }
                }
            }
            if model.hasStaleLoadedRuntime {
                Text("Switching between greedy and sampled decoding, or turning the repetition penalty on or off, requires a model reload.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Button("Reset to Defaults", action: model.resetGenerationParameters)
                .disabled(model.generationParametersAreDefault)
        } header: {
            Text("Generation")
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    /// Clamped so neither the field nor the stepper can store a length the
    /// settings file would reject, or one larger than the context itself.
    private var maxResponseTokensBinding: Binding<Int> {
        Binding(
            get: { model.maxResponseTokens },
            set: { model.maxResponseTokens = min(max($0, 16), model.maxContextTokens) })
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
            VStack(alignment: .leading, spacing: 8) {
                Text("RDADVISE")
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

}
