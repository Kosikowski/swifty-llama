import Foundation
import SLlama

/// A unified actor that combines generation coordination and core functionality
/// This solves the async stream context issues by keeping everything within the same actor context
@SwiftyLlamaActor
public class SwiftyCoreLlama {
    // MARK: - Private Properties

    private let model: SLlamaModel
    private let vocab: SLlamaVocab
    private let maxContextSize: Int32

    // Single context per instance (like the working example)
    private var context: SLlamaContext?
    private var isContextInitialized = false

    // Single batch instance (like the working example)
    private var batch: SLlamaBatch?

    // State tracking (like the working example)
    private var currentTokenCount: Int32 = 0
    private var shouldContinuePredicting = false
    private var tokenBuffer: [SLlamaToken] = []

    // Conversation management
    public struct Conversation: Codable {
        public let id: ConversationID
        public var messages: [ConversationMessage]
        public var totalTokens: Int32
        public let createdAt: Date

        public init(id: ConversationID, messages: [ConversationMessage], totalTokens: Int32, createdAt: Date) {
            self.id = id
            self.messages = messages
            self.totalTokens = totalTokens
            self.createdAt = createdAt
        }
    }

    public struct ConversationMessage: Codable {
        public let role: String
        public let content: String
        public let tokens: [SLlamaToken] // SLlamaToken == Int32 is Codable
        public let timestamp: Date

        public init(role: String, content: String, tokens: [SLlamaToken], timestamp: Date) {
            self.role = role
            self.content = content
            self.tokens = tokens
            self.timestamp = timestamp
        }
    }

    private var conversations: [ConversationID: Conversation] = [:]
    private var currentConversationId: ConversationID?

    // Generation tracking
    private struct LiveGeneration {
        let id: GenerationID
        let conversationId: ConversationID?
        var params: GenerationParams
        let startTime: Date
        var task: Task<Void, Never>?
    }

    private var activeGenerations: [GenerationID: LiveGeneration] = [:]

    // MARK: - Initialization

    public init(modelPath: String, maxCtx: Int32 = 2048) throws {
        SLlama.initialize()
        model = try SLlamaModel(modelPath: modelPath)
        vocab = SLlamaVocab(vocab: model.vocab)
        maxContextSize = maxCtx
    }

    // MARK: - Private Context Management

    private func ensureContextInitialized() throws {
        guard !isContextInitialized else { return }

        // Create context only once per instance
        context = try SLlamaContext(model: model)

        // Create single batch instance (like the working example)
        batch = SLlamaBatch(nTokens: maxContextSize, nSeqMax: 1)

        isContextInitialized = true
    }

    private func configureContext(with params: GenerationParams) throws {
        guard let context else { throw GenerationError.contextNotInitialized }

        // Configure the context using parameters
        context.setThreads(nThreads: params.threads, nThreadsBatch: params.batchThreads)
        context.setEmbeddings(params.enableEmbeddings)
        context.setCausalAttention(params.enableCausalAttention)
    }

    // MARK: - Conversation Management

    /// Start a new conversation
    public func startNewConversation() -> ConversationID {
        let id = ConversationID()
        conversations[id] = Conversation(
            id: id,
            messages: [],
            totalTokens: 0,
            createdAt: Date()
        )
        currentConversationId = id
        return id
    }

    /// Continue an existing conversation
    public func continueConversation(_ id: ConversationID) throws {
        guard conversations[id] != nil else {
            throw GenerationError.conversationNotFound
        }
        currentConversationId = id
    }

    /// Get current conversation ID
    public func getCurrentConversationId() -> ConversationID? {
        currentConversationId
    }

    /// Get conversation info
    public func getConversationInfo(_ id: ConversationID) -> ConversationInfo? {
        guard let conversation = conversations[id] else { return nil }

        return ConversationInfo(
            id: conversation.id,
            messageCount: conversation.messages.count,
            totalTokens: conversation.totalTokens,
            createdAt: conversation.createdAt
        )
    }

    /// Clear a conversation (removes context)
    public func clearConversation(_ id: ConversationID) {
        conversations.removeValue(forKey: id)
        if currentConversationId == id {
            currentConversationId = nil
        }
        // Reset context state
        currentTokenCount = 0
        tokenBuffer.removeAll()
        shouldContinuePredicting = false
    }

    // MARK: - Batch Management (like the working example)

    private func clearBatch() {
        guard let batch else { return }
        batch.clear()
    }

    private func addToBatch(token: SLlamaToken, position: Int32, isLogit: Bool = true) {
        guard let batch else { return }
        batch.addToken(token, position: position, sequenceIds: [0], logits: isLogit)
    }

    // MARK: - Persistence Support

    /// Get the current state of all conversations for persistence
    public func getConversationState() -> [Conversation] {
        Array(conversations.values)
    }

    /// Restore conversations from persisted state
    public func restoreConversations(_ savedConversations: [Conversation]) {
        conversations.removeAll()
        for conversation in savedConversations {
            conversations[conversation.id] = conversation
        }
    }

    /// Save conversations to JSON data
    public func saveConversationsToJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(getConversationState())
    }

    /// Load conversations from JSON data
    public func loadConversationsFromJSON(_ data: Data) throws {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let savedConversations = try decoder.decode([Conversation].self, from: data)
        restoreConversations(savedConversations)
    }

    /// Warm up context with conversation history (lazy reconstruction of KV cache)
    private func warmUpContext(with conversation: Conversation) throws {
        guard let ctx = context else { return }
        let hist = conversation.messages.flatMap(\.tokens)
        guard !hist.isEmpty else { return }

        // 1) Start from a clean slate
        ctx.clearMemory(data: true)
        currentTokenCount = 0
        tokenBuffer.removeAll()
        clearBatch()

        // 2) Configure context for warm-up (use default params)
        let defaultParams = GenerationParams(temperature: 0.7, maxTokens: 100)
        try configureContext(with: defaultParams)

        // 3) Feed every historical token
        for (i, tok) in hist.enumerated() {
            addToBatch(
                token: tok,
                position: Int32(i),
                isLogit: i == hist.count - 1
            ) // last token gets logits
        }
        try ctx.core().decode(batch!)
        currentTokenCount = Int32(hist.count)
    }

    /// Continue a conversation with warm-up (for restored conversations)
    public func continueConversationWithWarmUp(_ id: ConversationID) throws {
        guard let conversation = conversations[id] else {
            throw GenerationError.conversationNotFound
        }

        // Warm up the context with conversation history
        try warmUpContext(with: conversation)
        currentConversationId = id
    }

    /// Check if a conversation needs warm-up (has history but context is cold)
    private func needsWarmUp(for conversationId: ConversationID) -> Bool {
        guard let conversation = conversations[conversationId] else { return false }
        // Need warm-up if conversation has history but our context is empty
        // AND we haven't already warmed up this conversation
        return !conversation.messages.isEmpty && (currentTokenCount == 0 || tokenBuffer.isEmpty)
    }

    // MARK: - Public API

    /// Begin a new generation and immediately obtain a token stream
    /// If no conversation is active, starts a new one
    @discardableResult
    public func start(
        prompt: String,
        params: GenerationParams,
        conversationId: ConversationID? = nil
    ) async
        -> GenerationStream
    {
        let id = GenerationID()

        // Determine conversation to use
        let targetConversationId: ConversationID = if let conversationId {
            conversationId
        } else if let currentId = currentConversationId {
            currentId
        } else {
            startNewConversation()
        }

        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream(
            bufferingPolicy: .unbounded
        )

        // Start generation task within the same actor context - NO detached task as it would cause segmentation fault
        let task = Task { @SwiftyLlamaActor in
            await self.performGeneration(
                id: id,
                prompt: prompt,
                params: params,
                conversationId: targetConversationId,
                continuation: continuation
            )
            return ()
        }

        // Store generation info with the actual task
        activeGenerations[id] = LiveGeneration(
            id: id,
            conversationId: targetConversationId,
            params: params,
            startTime: Date(),
            task: task
        )

        // Set up termination callback
        continuation.onTermination = { @Sendable _ in
            Task { await self.cancel(id) }
        }

        return GenerationStream(id: id, stream: stream)
    }

    // MARK: - Private Generation Logic

    private func performGeneration(
        id _: GenerationID,
        prompt: String,
        params: GenerationParams,
        conversationId: ConversationID,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async {
        do {
            // Ensure context is initialized (only once per instance)
            try ensureContextInitialized()

            // Configure context for this generation (only if not already warmed up)
            if currentTokenCount == 0 {
                try configureContext(with: params)
            }

            // Tokenize prompt
            let promptTokens = try vocab.tokenize(text: prompt)

            // Prepare context with conversation history
            guard prepareContextWithConversation(
                for: promptTokens,
                conversationId: conversationId,
                llamaContext: context!
            ) else {
                continuation.finish(throwing: GenerationError.contextPreparationFailed)
                return
            }

            // Recreate sampler with updated context
            let sampler = createSampler(with: params, llamaContext: context!)
            sampler.reset()

            // Generate tokens
            var generatedTokens: [SLlamaToken] = []
            let maxNewTokens = max(0, Int32(params.maxTokens) - currentTokenCount)

            // If we already have more tokens than requested, we can still generate a minimum number
            let minNewTokens = maxNewTokens > 0 ? maxNewTokens : 1
            while shouldContinuePredicting, generatedTokens.count < minNewTokens {
                let token = predictNextToken(sampler: sampler, llamaContext: context!)

                if token == vocab.eosToken {
                    break
                }

                generatedTokens.append(token)
                let tokenText = try vocab.tokenToPiece(token: token)
                continuation.yield(tokenText)
            }

            // Store the conversation
            await storeConversationMessage(
                conversationId: conversationId,
                role: "user",
                content: prompt,
                tokens: promptTokens
            )

            if !generatedTokens.isEmpty {
                let responseContent = try generatedTokens.map { try vocab.tokenToPiece(token: $0) }.joined()
                await storeConversationMessage(
                    conversationId: conversationId,
                    role: "assistant",
                    content: responseContent,
                    tokens: generatedTokens
                )
            }

            continuation.finish()

        } catch {
            // Convert specific errors to GenerationError
            let generationError: GenerationError = if error is GenerationError {
                error as! GenerationError
            } else {
                // Map underlying errors to appropriate GenerationError cases
                switch error {
                    case _ where error.localizedDescription.contains("tokenize"):
                        .tokenizationFailed
                    case _ where error.localizedDescription.contains("context"):
                        .contextPreparationFailed
                    default:
                        .generationFailed
                }
            }
            continuation.finish(throwing: generationError)
        }
    }

    // MARK: - Context Preparation with Conversation History

    private func prepareContextWithConversation(
        for promptTokens: [SLlamaToken],
        conversationId: ConversationID,
        llamaContext: SLlamaContext
    )
        -> Bool
    {
        guard !promptTokens.isEmpty else { return false }

        // Check if conversation exists
        if let conversation = conversations[conversationId] {
            // Existing conversation - continue from history
            let historyTokens = conversation.messages.flatMap(\.tokens)
            let totalTokens = historyTokens.count + promptTokens.count

            guard maxContextSize > totalTokens else {
                // Context too long - need to truncate or start new conversation
                return false
            }

            // Note: Warm-up should be done before calling this method via continueConversationWithWarmUp
            // We don't warm up here to avoid clearing the context during generation

            // If we have history, we need to continue from where we left off
            if !historyTokens.isEmpty {
                currentTokenCount = Int32(historyTokens.count)
                tokenBuffer = historyTokens

                // For continuation, we need to start new tokens from the next position
                clearBatch()

                // Add new prompt tokens starting from the next position after history
                for (i, token) in promptTokens.enumerated() {
                    let position = Int32(historyTokens.count + i)
                    addToBatch(token: token, position: position, isLogit: i == promptTokens.count - 1)
                }
            } else {
                // Empty conversation - need to clear KV cache since we're starting fresh
                llamaContext.clearMemory(data: true)

                currentTokenCount = 0
                tokenBuffer.removeAll()

                clearBatch()

                // For new conversation, start from position 0 since we cleared the cache
                for (i, token) in promptTokens.enumerated() {
                    addToBatch(token: token, position: Int32(i), isLogit: i == promptTokens.count - 1)
                }
            }

            do {
                try llamaContext.core().decode(batch!)
            } catch {
                return false
            }

            // Update token count based on whether we have history or not
            if !historyTokens.isEmpty {
                currentTokenCount = Int32(historyTokens.count + promptTokens.count)
            } else {
                currentTokenCount = Int32(promptTokens.count)
            }
            shouldContinuePredicting = true
            return true
        } else {
            // New conversation - start fresh
            currentTokenCount = 0
            tokenBuffer.removeAll()

            // For new conversation, we need to reset the context completely
            // Reset our state
            currentTokenCount = 0
            tokenBuffer.removeAll()

            // Clear the KV cache completely for a brand-new conversation
            llamaContext.clearMemory(data: true)

            // Recreate the batch
            batch = SLlamaBatch(nTokens: maxContextSize, nSeqMax: 1)

            clearBatch()

            // For new conversation, start from position 0 since we cleared the cache
            for (i, token) in promptTokens.enumerated() {
                addToBatch(token: token, position: Int32(i), isLogit: i == promptTokens.count - 1)
            }

            do {
                try llamaContext.core().decode(batch!)
            } catch {
                return false
            }

            currentTokenCount = Int32(promptTokens.count)
            shouldContinuePredicting = true
            return true
        }
    }

    // MARK: - Context Preparation (like the working example) - for new conversations

    private func prepareContext(for promptTokens: [SLlamaToken], llamaContext _: SLlamaContext) -> Bool {
        guard !promptTokens.isEmpty else { return false }

        currentTokenCount = 0
        tokenBuffer.removeAll()

        let initialCount = promptTokens.count
        guard maxContextSize > initialCount else { return false }

        clearBatch()

        for (i, token) in promptTokens.enumerated() {
            addToBatch(token: token, position: Int32(i), isLogit: i == initialCount - 1)
        }

        do {
            try context!.core().decode(batch!)
        } catch {
            return false
        }

        currentTokenCount = Int32(initialCount)
        shouldContinuePredicting = true
        return true
    }

    // MARK: - Conversation Storage

    private func storeConversationMessage(
        conversationId: ConversationID,
        role: String,
        content: String,
        tokens: [SLlamaToken]
    ) async {
        // Ensure conversation exists
        if conversations[conversationId] == nil {
            conversations[conversationId] = Conversation(
                id: conversationId,
                messages: [],
                totalTokens: 0,
                createdAt: Date()
            )
        }

        guard var conversation = conversations[conversationId] else { return }

        let message = ConversationMessage(
            role: role,
            content: content,
            tokens: tokens,
            timestamp: Date()
        )

        conversation.messages.append(message)
        conversation.totalTokens += Int32(tokens.count)
        conversations[conversationId] = conversation
    }

    // MARK: - Token Prediction (like the working example)

    private func predictNextToken(sampler: SLlamaSampler, llamaContext _: SLlamaContext) -> SLlamaToken {
        guard shouldContinuePredicting, currentTokenCount < maxContextSize else {
            return vocab.eosToken
        }

        // Sample next token
        guard let token = sampler.sample() else {
            shouldContinuePredicting = false
            return vocab.eosToken
        }

        // Accept the token (like the working example)
        sampler.accept(token)

        tokenBuffer.append(token)

        // Check for stop conditions
        if token == vocab.eosToken {
            shouldContinuePredicting = false
            return vocab.eosToken
        }

        // Process the token for next generation
        clearBatch()
        addToBatch(token: token, position: currentTokenCount)

        do {
            try context!.core().decode(batch!)
        } catch {
            shouldContinuePredicting = false
            return vocab.eosToken
        }

        currentTokenCount += 1
        return token
    }

    private func createSampler(with params: GenerationParams, llamaContext: SLlamaContext) -> SLlamaSampler {
        // Create a temperature sampler with the given parameters
        let sampler = SLlamaSampler.temperature(
            context: llamaContext,
            temperature: params.temperature
        ) ?? SLlamaSampler.greedy(context: llamaContext) ?? SLlamaSampler(context: llamaContext)

        return sampler
    }

    // MARK: - Public Management Methods

    public func update(id: GenerationID, _ params: GenerationParams) async {
        if var generation = activeGenerations[id] {
            generation.params = params
            activeGenerations[id] = generation
        }
    }

    public func cancel(_ id: GenerationID) async {
        if let generation = activeGenerations[id] {
            generation.task?.cancel()
            activeGenerations.removeValue(forKey: id)
        }
    }

    public func getGenerationInfo(_ id: GenerationID) async -> GenerationInfo? {
        guard let generation = activeGenerations[id] else { return nil }

        return GenerationInfo(
            id: generation.id,
            conversationId: generation.conversationId,
            params: generation.params,
            startTime: generation.startTime,
            isActive: generation.task?.isCancelled == false
        )
    }

    public func getActiveGenerationIDs() async -> [GenerationID] {
        Array(activeGenerations.keys)
    }

    public func cancelAll() async {
        for (_, generation) in activeGenerations {
            generation.task?.cancel()
        }
        activeGenerations.removeAll()
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
