import Foundation
import SLlama

// MARK: - Shared Type Definitions

/// A unique identifier for a generation
public struct GenerationID: Hashable, Sendable {
    private let id = UUID()
}

/// Parameters for text generation
public struct GenerationParams: Hashable, Sendable {
    public let seed: UInt32
    public let topK: Int32
    public let topP: Float
    public let temperature: Float
    public let repeatPenalty: Float
    public let repetitionLookback: Int32
    public let maxTokens: Int32
    public let threads: Int32
    public let batchThreads: Int32
    public let enableEmbeddings: Bool
    public let enableCausalAttention: Bool

    public init(
        seed: UInt32 = 42,
        topK: Int32 = 40,
        topP: Float = 0.9,
        temperature: Float = 0.7,
        repeatPenalty: Float = 1.1,
        repetitionLookback: Int32 = 64,
        maxTokens: Int32 = 100,
        threads: Int32 = 4,
        batchThreads: Int32 = 4,
        enableEmbeddings: Bool = false,
        enableCausalAttention: Bool = true
    ) {
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.temperature = temperature
        self.repeatPenalty = repeatPenalty
        self.repetitionLookback = repetitionLookback
        self.maxTokens = maxTokens
        self.threads = threads
        self.batchThreads = batchThreads
        self.enableEmbeddings = enableEmbeddings
        self.enableCausalAttention = enableCausalAttention
    }
}

/// A stream of generated text tokens
public struct GenerationStream {
    public let id: GenerationID
    public let stream: AsyncThrowingStream<String, Error>

    public init(id: GenerationID, stream: AsyncThrowingStream<String, Error>) {
        self.id = id
        self.stream = stream
    }
}

/// Errors that can occur during generation
public enum GenerationError: Error, LocalizedError {
    case abortedByUser
    case modelNotLoaded
    case contextInitializationFailed
    case tokenizationFailed
    case generationFailed

    public var errorDescription: String? {
        switch self {
            case .abortedByUser:
                "Generation was aborted by user"
            case .modelNotLoaded:
                "Model is not loaded"
            case .contextInitializationFailed:
                "Failed to initialize context"
            case .tokenizationFailed:
                "Failed to tokenize input"
            case .generationFailed:
                "Generation failed"
        }
    }
}

/// A unique identifier for a conversation
public struct ConversationID: Hashable, Sendable {
    private let id = UUID()

    public init() {
        // Initialize with a new UUID
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    public static func == (lhs: ConversationID, rhs: ConversationID) -> Bool {
        lhs.id == rhs.id
    }
}

/// Information about a generation
public struct GenerationInfo {
    public let id: GenerationID
    public let conversationId: ConversationID?
    public let params: GenerationParams
    public let startTime: Date
    public let isActive: Bool
}

/// Information about a conversation
public struct ConversationInfo {
    public let id: ConversationID
    public let messageCount: Int
    public let totalTokens: Int32
    public let createdAt: Date
}

/// Information about a model
public struct ModelInfo {
    public let name: String
    public let contextSize: Int32
    public let vocabSize: Int32
}

/// Information about vocabulary
public struct VocabInfo {
    public let size: Int32
    public let bosToken: SLlamaToken
    public let eosToken: SLlamaToken
    public let nlToken: SLlamaToken
}
