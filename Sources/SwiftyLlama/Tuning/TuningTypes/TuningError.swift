

import Foundation

/// Errors that can occur during fine-tuning operations
public enum TuningError: Error, LocalizedError, Equatable {
    case contextNotInitialized
    case modelNotLoaded
    case tokenizerNotInitialized
    case adapterFileNotFound(path: String)
    case adapterApplicationFailed(path: String, errorDescription: String)
    case invalidLoRARank(rank: Int)
    case invalidLearningRate(rate: Float)
    case invalidEpochs(epochs: Int)
    case trainingSessionNotFound
    case incompatibleAdapter

    public var errorDescription: String? {
        switch self {
            case .contextNotInitialized:
                "Context is not initialized"
            case .modelNotLoaded:
                "Model is not loaded"
            case .tokenizerNotInitialized:
                "Tokenizer is not initialized"
            case let .adapterFileNotFound(path):
                "LoRA adapter file not found at path: \(path)"
            case let .adapterApplicationFailed(path, errorDescription):
                "Failed to apply LoRA adapter at path: \(path), error: \(errorDescription)"
            case let .invalidLoRARank(rank):
                "Invalid LoRA rank: \(rank). Must be between 1 and 128"
            case let .invalidLearningRate(rate):
                "Invalid learning rate: \(rate). Must be between 0 and 1"
            case let .invalidEpochs(epochs):
                "Invalid number of epochs: \(epochs). Must be between 1 and 100"
            case .trainingSessionNotFound:
                "No active training session found"
            case .incompatibleAdapter:
                "LoRA adapter is incompatible with current model"
        }
    }
}
