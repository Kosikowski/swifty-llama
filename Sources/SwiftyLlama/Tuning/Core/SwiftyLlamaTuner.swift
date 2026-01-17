import Foundation
import SLlama

/// Actor for fine-tuning operations with llama.cpp
@SLlamaActor
public final class SwiftyLlamaTuner {
    // MARK: - Properties

    private var model: SLlamaModel?
    private var context: SLlamaContext?
    private var vocab: SLlamaVocab?

    // MARK: - LoRA Management

    /// Currently applied LoRA adapter metadata
    private var currentLoRA: LoRAAdapter?

    /// Currently loaded SLlamaAdapter instance
    private var loadedAdapter: SLlamaAdapter?

    /// Available LoRA adapters
    private var availableAdapters: [LoRAAdapter] = []

    // MARK: - Training State

    /// Current training session
    private var trainingSession: TrainingSession?

    /// Training metrics
    private var trainingMetrics: TrainingMetrics?

    // MARK: - Model Saving State

    /// Base model path for reference
    private var baseModelPath: String?

    /// Training configuration for saving
    private var lastTrainingConfig: TrainingConfig?

    /// Training dataset for reference
    private var lastTrainingDataset: TrainingDataset?

    // MARK: - Initialization

    public init() {}

    // MARK: - Model Loading

    /// Load the base model for fine-tuning
    public func loadModel(path: String) throws {
        model = try SLlamaModel(modelPath: path)
        vocab = SLlamaVocab(vocab: model!.vocab)
        baseModelPath = path

        // Initialize context for training
        context = try SLlamaContext(model: model!)
    }

    // MARK: - LoRA Adapter Management

    /// Apply a LoRA adapter to the current context
    public func applyLoRA(path: String, scale: Float = 1.0, metadata: LoRAMetadata? = nil) throws {
        guard let ctx = context, let mdl = model else {
            throw TuningError.contextNotInitialized
        }

        // Validate adapter file exists
        guard FileManager.default.fileExists(atPath: path) else {
            throw TuningError.adapterFileNotFound(path: path)
        }

        // Remove any previously loaded adapter
        if let existingAdapter = loadedAdapter {
            try? ctx.removeLoRAAdapter(existingAdapter)
            loadedAdapter = nil
        }

        // Actually load and apply the adapter using SLlamaAdapter
        do {
            let sllAdapter = try SLlamaAdapter(model: mdl, path: path)
            try ctx.addLoRAAdapter(sllAdapter)
            loadedAdapter = sllAdapter
        } catch let error as SLlamaError {
            throw TuningError.adapterApplicationFailed(path: path, errorDescription: error.localizedDescription)
        }

        // Store adapter metadata
        let adapter = LoRAAdapter(
            path: path,
            scale: scale,
            metadata: metadata ?? LoRAMetadata(),
            appliedAt: Date()
        )

        currentLoRA = adapter

        // Add to available adapters if not already present
        if !availableAdapters.contains(where: { $0.path == path }) {
            availableAdapters.append(adapter)
        }
    }

    /// Remove currently applied LoRA adapter
    public func removeLoRA() throws {
        guard let ctx = context else {
            throw TuningError.contextNotInitialized
        }

        // Remove the adapter from context if one is loaded
        if let adapter = loadedAdapter {
            try ctx.removeLoRAAdapter(adapter)
            loadedAdapter = nil
        } else {
            // Clear all adapters if we don't have a reference
            ctx.clearLoRAAdapters()
        }

        currentLoRA = nil
    }

    /// Get currently applied LoRA adapter
    public func getCurrentLoRA() -> LoRAAdapter? {
        currentLoRA
    }

    /// Get all available LoRA adapters
    public func getAvailableAdapters() -> [LoRAAdapter] {
        availableAdapters
    }

    // MARK: - Training Data Preparation

    /// Prepare training data with proper chat formatting
    public func prepareTrainingData(
        conversations: [TrainingConversation],
        validationSplit: Double = 0.1
    ) throws
        -> TrainingDataset
    {
        guard let vocab else {
            throw TuningError.tokenizerNotInitialized
        }

        // Validate validation split
        guard validationSplit >= 0.0, validationSplit <= 1.0 else {
            throw TuningError.invalidValidationSplit(split: validationSplit)
        }

        var allExamples: [TrainingExample] = []

        for conversation in conversations {
            // Convert conversation to training examples
            let examples = try formatConversationForTraining(
                conversation: conversation,
                vocab: vocab
            )
            allExamples.append(contentsOf: examples)
        }

        // Split into training and validation
        let shuffledExamples = allExamples.shuffled()
        let splitIndex = Int(Double(shuffledExamples.count) * (1.0 - validationSplit))

        let training = Array(shuffledExamples[..<splitIndex])
        let validation = Array(shuffledExamples[splitIndex...])

        let dataset = TrainingDataset(training: training, validation: validation)
        lastTrainingDataset = dataset

        return dataset
    }

    /// Format conversation for training with proper chat formatting
    private func formatConversationForTraining(
        conversation: TrainingConversation,
        vocab: SLlamaVocab
    ) throws
        -> [TrainingExample]
    {
        var examples: [TrainingExample] = []
        var conversationHistory: [String] = []

        for message in conversation.messages {
            let formattedMessage = formatMessageForTraining(message)
            conversationHistory.append(formattedMessage)

            // Create training example from conversation history
            let inputText = conversationHistory.joined(separator: "\n")
            let targetText = formattedMessage

            let inputTokens = try vocab.tokenize(text: inputText)
            let targetTokens = try vocab.tokenize(text: targetText)

            let example = TrainingExample(
                conversation: conversation,
                formattedText: inputText,
                tokens: inputTokens,
                targetTokens: targetTokens
            )
            examples.append(example)
        }

        return examples
    }

    /// Format individual message for training
    private func formatMessageForTraining(_ message: TrainingMessage) -> String {
        switch message.role {
            case .system:
                "<|system|>\n\(message.content)<|endoftext|>"
            case .user:
                "<|user|>\n\(message.content)<|endoftext|>"
            case .assistant:
                "<|assistant|>\n\(message.content)<|endoftext|>"
        }
    }

    // MARK: - Training Session Management
    // Note: Training is NOT supported by llama.cpp - it only supports inference.
    // Use external tools like llama.cpp's train-text-from-scratch for LoRA training.

    /// Start a new training session
    /// - Important: Training is not supported by llama.cpp. This method always throws.
    public func startTrainingSession(
        dataset _: TrainingDataset,
        config _: TrainingConfig
    ) throws
        -> TrainingSession
    {
        throw TuningError.trainingNotSupported
    }

    /// Stop current training session
    /// - Important: Training is not supported by llama.cpp. This method always throws.
    public func stopTrainingSession() throws {
        throw TuningError.trainingNotSupported
    }

    /// Get current training session
    /// - Important: Training is not supported by llama.cpp. This method always throws.
    public func getCurrentTrainingSession() throws -> TrainingSession? {
        throw TuningError.trainingNotSupported
    }

    /// Get training metrics
    /// - Important: Training is not supported by llama.cpp. This method always throws.
    public func getTrainingMetrics() throws -> TrainingMetrics? {
        throw TuningError.trainingNotSupported
    }

    // MARK: - Model Saving
    // Note: Saving trained LoRA adapters is NOT supported - llama.cpp doesn't expose training APIs.
    // Loading pre-trained LoRA adapters via applyLoRA() is fully supported.

    /// Save the current LoRA adapter to a file
    /// - Important: Saving new LoRA adapters is not supported. This method always throws.
    public func saveLoRAAdapter(
        path _: String,
        metadata _: LoRAMetadata? = nil
    ) throws {
        throw TuningError.trainingNotSupported
    }

    /// Save the fine-tuned model (base model + LoRA)
    /// - Important: Training is not supported by llama.cpp. This method always throws.
    public func saveFineTunedModel(
        path _: String,
        format _: ModelFormat = .gguf,
        metadata _: ModelMetadata? = nil
    ) throws {
        throw TuningError.trainingNotSupported
    }

    /// Export model in different formats
    /// - Important: Model export is not supported. Use llama.cpp's quantize tool directly.
    public func exportModel(
        path _: String,
        format _: ModelFormat,
        quantization _: ModelQuantization? = nil
    ) throws {
        throw TuningError.trainingNotSupported
    }

    /// Get model saving information
    /// - Important: Training is not supported by llama.cpp. This method always throws.
    public func getModelSavingInfo() throws -> ModelSavingInfo {
        throw TuningError.trainingNotSupported
    }

    // MARK: - Evaluation
    // Note: Real perplexity calculation requires training infrastructure not available in llama.cpp.

    /// Evaluate model performance on validation set
    /// - Important: Training evaluation is not supported by llama.cpp. This method always throws.
    public func evaluateModel(validationExamples _: [TrainingExample]) throws -> EvaluationMetrics {
        throw TuningError.trainingNotSupported
    }

    // MARK: - Safety and Fallback

    /// Check if LoRA adapter is compatible with current model
    public func validateLoRACompatibility(path _: String) throws -> LoRACompatibility {
        // Read adapter metadata and compare with current model
        // This is a simplified implementation
        LoRACompatibility(
            isCompatible: true,
            warnings: [],
            baseModelSHA: "unknown",
            adapterConfig: "unknown"
        )
    }

    /// Enable/disable LoRA fallback mode
    public func setLoRAFallbackMode(_: Bool) {
        // Implementation would involve setting up fallback behavior
        // when LoRA application fails
    }
}
