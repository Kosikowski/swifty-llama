import Foundation

/// Public protocol for fine-tuning operations with llama.cpp
/// This protocol defines the interface for SwiftyTuningLlama functionality
@SwiftyLlamaActor
public protocol SwiftyLlamaTuning: AnyObject {
    // MARK: - Model Loading

    /// Load the base model for fine-tuning
    /// - Parameter path: Path to the model file
    /// - Throws: TuningError if model loading fails
    func loadModel(path: String) throws

    // MARK: - LoRA Adapter Management

    /// Apply a LoRA adapter to the current context
    /// - Parameters:
    ///   - path: Path to the LoRA adapter file
    ///   - scale: Scaling factor for the adapter (default: 1.0)
    ///   - metadata: Optional metadata for the adapter
    /// - Throws: TuningError if adapter application fails
    func applyLoRA(path: String, scale: Float, metadata: LoRAMetadata?) throws

    /// Remove currently applied LoRA adapter
    /// - Throws: TuningError if context is not initialized
    func removeLoRA() throws

    /// Get currently applied LoRA adapter
    /// - Returns: The currently applied LoRA adapter or nil if none is applied
    func getCurrentLoRA() -> LoRAAdapter?

    /// Get all available LoRA adapters
    /// - Returns: Array of available LoRA adapters
    func getAvailableAdapters() -> [LoRAAdapter]

    // MARK: - Training Data Preparation

    /// Prepare training data with proper chat formatting
    /// - Parameters:
    ///   - conversations: Array of training conversations
    ///   - validationSplit: Fraction of data to use for validation (default: 0.1)
    /// - Returns: TrainingDataset with training and validation examples
    /// - Throws: TuningError if tokenizer is not initialized
    func prepareTrainingData(
        conversations: [TrainingConversation],
        validationSplit: Double
    ) throws
        -> TrainingDataset

    // MARK: - Training Session Management

    /// Start a new training session
    /// - Parameters:
    ///   - dataset: Training dataset with training and validation examples
    ///   - config: Training configuration parameters
    /// - Returns: TrainingSession with session information
    /// - Throws: TuningError if context is not initialized or config is invalid
    func startTrainingSession(
        dataset: TrainingDataset,
        config: TrainingConfig
    ) throws
        -> TrainingSession

    /// Stop current training session
    func stopTrainingSession()

    /// Get current training session
    /// - Returns: Current training session or nil if no session is active
    func getCurrentTrainingSession() -> TrainingSession?

    /// Get training metrics
    /// - Returns: Current training metrics or nil if no session is active
    func getTrainingMetrics() -> TrainingMetrics?

    // MARK: - Model Saving

    /// Save the current LoRA adapter to a file
    /// - Parameters:
    ///   - path: Output path for the LoRA adapter
    ///   - metadata: Additional metadata to include
    /// - Throws: TuningError if saving fails
    func saveLoRAAdapter(path: String, metadata: LoRAMetadata?) throws

    /// Save the fine-tuned model (base model + LoRA)
    /// - Parameters:
    ///   - path: Output path for the fine-tuned model
    ///   - format: Model format (gguf, safetensors, etc.)
    ///   - metadata: Additional metadata to include
    /// - Throws: TuningError if saving fails
    func saveFineTunedModel(
        path: String,
        format: ModelFormat,
        metadata: ModelMetadata?
    ) throws

    /// Export model in different formats
    /// - Parameters:
    ///   - path: Output path
    ///   - format: Target format
    ///   - quantization: Quantization settings
    /// - Throws: TuningError if export fails
    func exportModel(
        path: String,
        format: ModelFormat,
        quantization: ModelQuantization?
    ) throws

    /// Get model saving information
    /// - Returns: ModelSavingInfo with current state
    func getModelSavingInfo() -> ModelSavingInfo

    // MARK: - Evaluation

    /// Evaluate model performance on validation set
    /// - Parameter validationExamples: Array of validation examples
    /// - Returns: EvaluationMetrics with perplexity, loss, and token counts
    /// - Throws: TuningError if context is not initialized
    func evaluateModel(validationExamples: [TrainingExample]) throws -> EvaluationMetrics

    // MARK: - Safety and Fallback

    /// Check if LoRA adapter is compatible with current model
    /// - Parameter path: Path to the LoRA adapter file
    /// - Returns: LoRACompatibility information
    /// - Throws: TuningError if compatibility check fails
    func validateLoRACompatibility(path: String) throws -> LoRACompatibility

    /// Enable/disable LoRA fallback mode
    /// - Parameter enabled: Whether to enable fallback mode
    func setLoRAFallbackMode(_ enabled: Bool)
}

// MARK: - Default Parameter Extensions

public extension SwiftyLlamaTuning {
    /// Apply a LoRA adapter with default scale
    /// - Parameters:
    ///   - path: Path to the LoRA adapter file
    ///   - metadata: Optional metadata for the adapter
    /// - Throws: TuningError if adapter application fails
    func applyLoRA(path: String, metadata: LoRAMetadata? = nil) throws {
        try applyLoRA(path: path, scale: 1.0, metadata: metadata)
    }

    /// Save LoRA adapter with default metadata
    /// - Parameter path: Output path for the LoRA adapter
    /// - Throws: TuningError if saving fails
    func saveLoRAAdapter(path: String) throws {
        try saveLoRAAdapter(path: path, metadata: nil)
    }

    /// Save fine-tuned model with default format and metadata
    /// - Parameter path: Output path for the fine-tuned model
    /// - Throws: TuningError if saving fails
    func saveFineTunedModel(path: String) throws {
        try saveFineTunedModel(
            path: path,
            format: .gguf,
            metadata: nil
        )
    }

    /// Export model with default quantization
    /// - Parameters:
    ///   - path: Output path
    ///   - format: Target format
    /// - Throws: TuningError if export fails
    func exportModel(path: String, format: ModelFormat) throws {
        try exportModel(path: path, format: format, quantization: nil)
    }
}

// MARK: - Protocol Conformance

/// Make SwiftyTuningLlama conform to the protocol
extension SwiftyLlamaTuner: SwiftyLlamaTuning {}
