import Foundation

/// Public protocol for fine-tuning operations with llama.cpp
/// This protocol defines the interface for SwiftyTuningLlama functionality
///
/// ## Supported Operations
/// - Loading base models
/// - Loading and applying pre-trained LoRA adapters
/// - Removing LoRA adapters
/// - Preparing training data (tokenization)
///
/// ## Unsupported Operations (will throw TuningError.trainingNotSupported)
/// - Training sessions (llama.cpp only supports inference)
/// - Saving new LoRA adapters
/// - Saving fine-tuned models
/// - Model export
/// - Training evaluation
///
/// Use external tools for LoRA training.
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
    // Note: Training is NOT supported by llama.cpp - it only supports inference.
    // These methods will throw TuningError.trainingNotSupported when called.
    // Use external tools like llama.cpp's train-text-from-scratch for LoRA training.

    /// Start a new training session
    /// - Important: Training is not supported by llama.cpp. This method will throw TuningError.trainingNotSupported.
    /// - Parameters:
    ///   - dataset: Training dataset with training and validation examples
    ///   - config: Training configuration parameters
    /// - Returns: TrainingSession with session information
    /// - Throws: TuningError.trainingNotSupported - always thrown as training is not supported
    func startTrainingSession(
        dataset: TrainingDataset,
        config: TrainingConfig
    ) throws
        -> TrainingSession

    /// Stop current training session
    /// - Important: Training is not supported by llama.cpp. This method will throw TuningError.trainingNotSupported.
    /// - Throws: TuningError.trainingNotSupported
    func stopTrainingSession() throws

    /// Get current training session
    /// - Important: Training is not supported by llama.cpp. This method will throw TuningError.trainingNotSupported.
    /// - Returns: Current training session or nil if no session is active
    /// - Throws: TuningError.trainingNotSupported
    func getCurrentTrainingSession() throws -> TrainingSession?

    /// Get training metrics
    /// - Important: Training is not supported by llama.cpp. This method will throw TuningError.trainingNotSupported.
    /// - Returns: Current training metrics or nil if no session is active
    /// - Throws: TuningError.trainingNotSupported
    func getTrainingMetrics() throws -> TrainingMetrics?

    // MARK: - Model Saving
    // Note: Saving trained LoRA adapters is NOT supported - llama.cpp doesn't expose training APIs.
    // Loading pre-trained LoRA adapters via applyLoRA() is fully supported.
    // These methods will throw TuningError.trainingNotSupported when called.

    /// Save the current LoRA adapter to a file
    /// - Important: Saving new LoRA adapters is not supported. This method will throw TuningError.trainingNotSupported.
    /// - Parameters:
    ///   - path: Output path for the LoRA adapter
    ///   - metadata: Additional metadata to include
    /// - Throws: TuningError.trainingNotSupported
    func saveLoRAAdapter(path: String, metadata: LoRAMetadata?) throws

    /// Save the fine-tuned model (base model + LoRA)
    /// - Important: Training is not supported by llama.cpp. This method will throw TuningError.trainingNotSupported.
    /// - Parameters:
    ///   - path: Output path for the fine-tuned model
    ///   - format: Model format (gguf, safetensors, etc.)
    ///   - metadata: Additional metadata to include
    /// - Throws: TuningError.trainingNotSupported
    func saveFineTunedModel(
        path: String,
        format: ModelFormat,
        metadata: ModelMetadata?
    ) throws

    /// Export model in different formats
    /// - Important: Model export is not supported. Use llama.cpp's quantize tool directly.
    /// - Parameters:
    ///   - path: Output path
    ///   - format: Target format
    ///   - quantization: Quantization settings
    /// - Throws: TuningError.trainingNotSupported
    func exportModel(
        path: String,
        format: ModelFormat,
        quantization: ModelQuantization?
    ) throws

    /// Get model saving information
    /// - Important: Training is not supported by llama.cpp. This method will throw TuningError.trainingNotSupported.
    /// - Returns: ModelSavingInfo with current state
    /// - Throws: TuningError.trainingNotSupported
    func getModelSavingInfo() throws -> ModelSavingInfo

    // MARK: - Evaluation
    // Note: Real perplexity calculation requires training infrastructure not available in llama.cpp.

    /// Evaluate model performance on validation set
    /// - Important: Training evaluation is not supported by llama.cpp. This method will throw
    /// TuningError.trainingNotSupported.
    /// - Parameter validationExamples: Array of validation examples
    /// - Returns: EvaluationMetrics with perplexity, loss, and token counts
    /// - Throws: TuningError.trainingNotSupported
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
    /// - Important: Saving new LoRA adapters is not supported. This method will throw TuningError.trainingNotSupported.
    /// - Parameter path: Output path for the LoRA adapter
    /// - Throws: TuningError.trainingNotSupported
    func saveLoRAAdapter(path: String) throws {
        try saveLoRAAdapter(path: path, metadata: nil)
    }

    /// Save fine-tuned model with default format and metadata
    /// - Important: Training is not supported. This method will throw TuningError.trainingNotSupported.
    /// - Parameter path: Output path for the fine-tuned model
    /// - Throws: TuningError.trainingNotSupported
    func saveFineTunedModel(path: String) throws {
        try saveFineTunedModel(
            path: path,
            format: .gguf,
            metadata: nil
        )
    }

    /// Export model with default quantization
    /// - Important: Model export is not supported. Use llama.cpp's quantize tool directly.
    /// - Parameters:
    ///   - path: Output path
    ///   - format: Target format
    /// - Throws: TuningError.trainingNotSupported
    func exportModel(path: String, format: ModelFormat) throws {
        try exportModel(path: path, format: format, quantization: nil)
    }
}

// MARK: - Protocol Conformance

/// Make SwiftyTuningLlama conform to the protocol
extension SwiftyLlamaTuner: SwiftyLlamaTuning {}
