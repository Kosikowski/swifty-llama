import Foundation
import SLlama

public typealias SwiftyLlamaActor = SLlamaActor

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

/// A unique identifier for a conversation
public struct ConversationID: Hashable, Sendable, Codable {
    private let id: UUID

    public init() {
        // Initialize with a new UUID
        id = UUID()
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    public static func == (lhs: ConversationID, rhs: ConversationID) -> Bool {
        lhs.id == rhs.id
    }
}

// MARK: - Fine-tuning Types

/// LoRA adapter information
public struct LoRAAdapter: Codable, Equatable, Sendable {
    public let path: String
    public let scale: Float
    public let metadata: LoRAMetadata
    public let appliedAt: Date

    public init(path: String, scale: Float, metadata: LoRAMetadata, appliedAt: Date) {
        self.path = path
        self.scale = scale
        self.metadata = metadata
        self.appliedAt = appliedAt
    }
}

/// LoRA metadata
public struct LoRAMetadata: Codable, Equatable, Sendable {
    public let baseModelSHA: String
    public let loraConfig: String
    public let trainingRunHash: String
    public let trainingDate: Date
    public let rank: Int
    public let alpha: Float

    public init(
        baseModelSHA: String = "unknown",
        loraConfig: String = "unknown",
        trainingRunHash: String = "unknown",
        trainingDate: Date = Date(),
        rank: Int = 8,
        alpha: Float = 16.0
    ) {
        self.baseModelSHA = baseModelSHA
        self.loraConfig = loraConfig
        self.trainingRunHash = trainingRunHash
        self.trainingDate = trainingDate
        self.rank = rank
        self.alpha = alpha
    }
}

/// Training conversation
public struct TrainingConversation: Codable, Sendable {
    public let id: String
    public let messages: [TrainingMessage]

    public init(id: String, messages: [TrainingMessage]) {
        self.id = id
        self.messages = messages
    }
}

/// Training message
public struct TrainingMessage: Codable, Sendable {
    public let role: MessageRole
    public let content: String

    public init(role: MessageRole, content: String) {
        self.role = role
        self.content = content
    }
}

/// Message role for training
public enum MessageRole: String, Codable, Sendable {
    case system
    case user
    case assistant
}

/// Training example
public struct TrainingExample: Codable, Sendable {
    public let conversation: TrainingConversation
    public let formattedText: String
    public let tokens: [SLlamaToken]

    public init(conversation: TrainingConversation, formattedText: String, tokens: [SLlamaToken]) {
        self.conversation = conversation
        self.formattedText = formattedText
        self.tokens = tokens
    }
}

/// Training dataset
public struct TrainingDataset: Codable, Sendable {
    public let training: [TrainingExample]
    public let validation: [TrainingExample]

    public init(training: [TrainingExample], validation: [TrainingExample]) {
        self.training = training
        self.validation = validation
    }
}

/// Training configuration
public struct TrainingConfig: Codable, Sendable {
    public let loraRank: Int
    public let learningRate: Float
    public let epochs: Int
    public let batchSize: Int
    public let useQLoRA: Bool
    public let qLoRAConfig: QLoRAConfig?

    public init(
        loraRank: Int = 8,
        learningRate: Float = 2e-5,
        epochs: Int = 3,
        batchSize: Int = 1,
        useQLoRA: Bool = false,
        qLoRAConfig: QLoRAConfig? = nil
    ) {
        self.loraRank = loraRank
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
        self.useQLoRA = useQLoRA
        self.qLoRAConfig = qLoRAConfig
    }
}

/// QLoRA configuration
public struct QLoRAConfig: Codable, Sendable {
    public let quantType: String
    public let useDoubleQuant: Bool
    public let computeDtype: String

    public init(
        quantType: String = "nf4",
        useDoubleQuant: Bool = true,
        computeDtype: String = "float16"
    ) {
        self.quantType = quantType
        self.useDoubleQuant = useDoubleQuant
        self.computeDtype = computeDtype
    }
}

/// Training session
public struct TrainingSession: Codable, Sendable {
    public let id: UUID
    public let dataset: TrainingDataset
    public let config: TrainingConfig
    public let startTime: Date
    public var endTime: Date?
    public var status: TrainingStatus

    public init(id: UUID, dataset: TrainingDataset, config: TrainingConfig, startTime: Date, status: TrainingStatus) {
        self.id = id
        self.dataset = dataset
        self.config = config
        self.startTime = startTime
        self.status = status
    }
}

/// Training status
public enum TrainingStatus: String, Codable, Sendable {
    case running
    case completed
    case stopped
    case failed
}

/// Training metrics
public struct TrainingMetrics: Codable, Sendable {
    public var currentEpoch: Int = 0
    public var currentStep: Int = 0
    public var trainingLoss: Float = 0.0
    public var validationLoss: Float = 0.0
    public var learningRate: Float = 0.0

    public init() {}
}

/// Evaluation metrics
public struct EvaluationMetrics: Codable, Sendable {
    public let perplexity: Float
    public let averageLoss: Float
    public let totalExamples: Int
    public let totalTokens: Int

    public init(perplexity: Float, averageLoss: Float, totalExamples: Int, totalTokens: Int) {
        self.perplexity = perplexity
        self.averageLoss = averageLoss
        self.totalExamples = totalExamples
        self.totalTokens = totalTokens
    }
}

/// LoRA compatibility information
public struct LoRACompatibility: Codable, Sendable {
    public let isCompatible: Bool
    public let warnings: [String]
    public let baseModelSHA: String
    public let adapterConfig: String

    public init(isCompatible: Bool, warnings: [String], baseModelSHA: String, adapterConfig: String) {
        self.isCompatible = isCompatible
        self.warnings = warnings
        self.baseModelSHA = baseModelSHA
        self.adapterConfig = adapterConfig
    }
}

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
