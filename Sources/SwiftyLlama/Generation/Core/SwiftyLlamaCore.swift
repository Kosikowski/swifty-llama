import Foundation
import SLlama

/// A unified actor that combines generation coordination and core functionality
/// This solves the async stream context issues by keeping everything within the same actor context
///
/// The implementation is split across multiple files for maintainability:
/// - `SwiftyLlamaCore.swift` - Core class, properties, initialization
/// - `SwiftyLlamaCore+Context.swift` - Context management and preparation
/// - `SwiftyLlamaCore+Conversation.swift` - Conversation management and persistence
/// - `SwiftyLlamaCore+Generation.swift` - Generation logic and token prediction
@SwiftyLlamaActor
public class SwiftyLlamaCore: SwiftyLlama {
    // MARK: - Internal Properties (accessible by extensions)

    let model: SLlamaModel
    let vocab: SLlamaVocab
    let maxContextSize: Int32

    // Shared context per instance (can be shared with proper memory clearing)
    var context: SLlamaContext?
    var isContextInitialized = false

    // Shared batch instance (can be shared with proper clearing)
    var batch: SLlamaBatch?

    // MARK: - Per-Conversation State

    struct ConversationState {
        // Token generation state
        var currentTokenCount: Int32 = 0
        var shouldContinuePredicting: Bool = false
        var tokenBuffer: [SLlamaToken] = []

        // Generation tracking
        var lastUsedParams: GenerationParams?
        var isActive: Bool = false
        var isWarmedUp: Bool = false
    }

    // Live generation tracking
    struct LiveGeneration {
        let id: GenerationID
        let conversationId: ConversationID
        var params: GenerationParams
        let startTime: Date
        let task: Task<Void, Never>?
    }

    // MARK: - Conversation and Generation State

    /// Active generations being tracked
    var activeGenerations: [GenerationID: LiveGeneration] = [:]

    /// Per-conversation state (token counts, warmup status, etc.)
    var conversationStates: [ConversationID: ConversationState] = [:]

    /// All conversations
    var conversations: [ConversationID: Conversation] = [:]

    /// Currently active conversation
    var currentConversationId: ConversationID?

    // MARK: - Public Properties

    /// Get the underlying model instance
    public var underlyingModel: SLlamaModel {
        model
    }

    // MARK: - Initialization

    public init(modelPath: String, contextSize: Int32 = 2048, backendType: SLlamaBackendType = .auto) throws {
        SLlama.initialize()
        model = try SLlamaModel(modelPath: modelPath, backendType: backendType)
        vocab = SLlamaVocab(model)
        maxContextSize = contextSize
    }

    // MARK: - Per-Conversation State Management

    func getOrCreateConversationState(for conversationId: ConversationID) -> ConversationState {
        if conversationStates[conversationId] == nil {
            conversationStates[conversationId] = ConversationState()
        }
        return conversationStates[conversationId]!
    }

    func updateConversationState(
        for conversationId: ConversationID,
        _ update: (inout ConversationState) -> Void
    ) {
        var state = getOrCreateConversationState(for: conversationId)
        update(&state)
        conversationStates[conversationId] = state
    }

    func clearConversationState(for conversationId: ConversationID) {
        conversationStates.removeValue(forKey: conversationId)
    }

    // MARK: - Convenience Extensions

    public var modelInfo: ModelInfo {
        ModelInfo(
            name: "Model",
            contextSize: maxContextSize,
            vocabSize: vocab.tokenCount
        )
    }

    public var vocabInfo: VocabInfo {
        VocabInfo(
            size: vocab.tokenCount,
            bosToken: vocab.bosToken,
            eosToken: vocab.eosToken,
            nlToken: vocab.nlToken
        )
    }
}
