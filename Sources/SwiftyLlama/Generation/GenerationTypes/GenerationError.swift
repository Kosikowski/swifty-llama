import Foundation

// Errors that can occur during generation
public enum GenerationError: Error, LocalizedError {
    case abortedByUser
    case modelNotLoaded
    case contextNotInitialized
    case conversationNotFound
    case contextPreparationFailed
    case tokenizationFailed
    case generationFailed
    case invalidState

    public var errorDescription: String? {
        switch self {
            case .abortedByUser:
                "Generation was aborted by user"
            case .modelNotLoaded:
                "Model is not loaded"
            case .contextNotInitialized:
                "Context not initialized"
            case .conversationNotFound:
                "Conversation not found"
            case .contextPreparationFailed:
                "Failed to prepare context"
            case .tokenizationFailed:
                "Failed to tokenize input"
            case .generationFailed:
                "Generation failed"
            case .invalidState:
                "Invalid state"
        }
    }
}
