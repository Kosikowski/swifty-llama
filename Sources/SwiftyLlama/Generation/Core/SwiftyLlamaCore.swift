import Foundation
import SLlama

/// A unified actor that combines generation coordination and core functionality
/// This solves the async stream context issues by keeping everything within the same actor context
@SwiftyLlamaActor
public class SwiftyLlamaCore: SwiftyLlama {
    // MARK: - Private Properties

    private let model: SLlamaModel
    private let vocab: SLlamaVocab
    private let maxContextSize: Int32

    // Shared context per instance (can be shared with proper memory clearing)
    private var context: SLlamaContext?
    private var isContextInitialized = false

    // Shared batch instance (can be shared with proper clearing)
    private var batch: SLlamaBatch?

    // Per-conversation state management
    private struct ConversationState {
        // Token generation state
        var currentTokenCount: Int32 = 0
        var shouldContinuePredicting: Bool = false
        var tokenBuffer: [SLlamaToken] = []

        // Generation tracking
        var lastUsedParams: GenerationParams?
        var isActive: Bool = false
        var isWarmedUp: Bool = false
    }

    private var conversationStates: [ConversationID: ConversationState] = [:]
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

    public init(modelPath: String, contextSize: Int32 = 2048) throws {
        SLlama.initialize()
        model = try SLlamaModel(modelPath: modelPath)
        vocab = SLlamaVocab(model)
        maxContextSize = contextSize
    }

    // MARK: - Private Context Management

    private func ensureContextInitialized() throws {
        guard !isContextInitialized else { return }

        // Create context only once per instance
        context = try SLlamaContext(model: model)

        // Create single batch instance
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

    // MARK: - Per-Conversation State Management

    private func getOrCreateConversationState(for conversationId: ConversationID) -> ConversationState {
        if conversationStates[conversationId] == nil {
            conversationStates[conversationId] = ConversationState()
        }
        return conversationStates[conversationId]!
    }

    private func updateConversationState(
        for conversationId: ConversationID,
        _ update: (inout ConversationState) -> Void
    ) {
        var state = getOrCreateConversationState(for: conversationId)
        update(&state)
        conversationStates[conversationId] = state
    }

    private func clearConversationState(for conversationId: ConversationID) {
        conversationStates.removeValue(forKey: conversationId)
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

        // Initialize conversation state
        _ = getOrCreateConversationState(for: id)

        return id
    }

    /// Continue an existing conversation
    public func continueConversation(_ id: ConversationID) throws {
        guard conversations[id] != nil else {
            throw GenerationError.conversationNotFound(conversationId: id)
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
        clearConversationState(for: id)

        if currentConversationId == id {
            currentConversationId = nil
        }
    }

    // MARK: - Batch Management
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
        conversationStates.removeAll()

        for conversation in savedConversations {
            conversations[conversation.id] = conversation
            // Initialize conversation state for restored conversations
            // Mark as not warmed up since we need to reconstruct the context
            updateConversationState(for: conversation.id) { state in
                state.isWarmedUp = false
                state.currentTokenCount = 0
                state.tokenBuffer.removeAll()
            }
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

        // Force a complete context reset by recreating the context entirely
        // Recreate the context to ensure complete reset
        context = try SLlamaContext(model: model)

        // Reinitialize the batch to ensure no residual state
        batch = SLlamaBatch(nTokens: maxContextSize, nSeqMax: 1)
        clearBatch()

        // Reset conversation states to ensure clean slate
        conversationStates.removeAll()

        restoreConversations(savedConversations)
    }

    /// Warm up context with conversation history (lazy reconstruction of KV cache)
    private func warmUpContext(with conversation: Conversation) throws {
        guard let ctx = context else { return }
        let hist = conversation.messages.flatMap(\.tokens)
        guard !hist.isEmpty else { return }

        // 1) Start from a clean slate
        ctx.clearMemory(data: true)

        // Update conversation state
        updateConversationState(for: conversation.id) { state in
            state.currentTokenCount = 0
            state.tokenBuffer.removeAll()
            state.isWarmedUp = true
        }

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

        // Update conversation state with historical token count
        updateConversationState(for: conversation.id) { state in
            state.currentTokenCount = Int32(hist.count)
            state.tokenBuffer = hist
        }
    }

    /// Continue a conversation with warm-up (for restored conversations)
    public func continueConversationWithWarmUp(_ id: ConversationID) throws {
        guard let conversation = conversations[id] else {
            throw GenerationError.conversationNotFound(conversationId: id)
        }

        // Force a complete context reset by recreating the context entirely
        // Recreate the context to ensure complete reset
        context = try SLlamaContext(model: model)

        // Reinitialize the batch to ensure no residual state
        batch = SLlamaBatch(nTokens: maxContextSize, nSeqMax: 1)
        clearBatch()

        // Reset conversation state to ensure clean slate
        updateConversationState(for: id) { state in
            state.currentTokenCount = 0
            state.tokenBuffer.removeAll()
            state.isWarmedUp = false
        }

        // Warm up the context with conversation history
        try warmUpContext(with: conversation)
        currentConversationId = id
    }

    /// Check if a conversation needs warm-up (has history but context is cold)
    private func needsWarmUp(for conversationId: ConversationID) -> Bool {
        guard let conversation = conversations[conversationId] else { return false }
        let state = getOrCreateConversationState(for: conversationId)

        // Need warm-up if conversation has history but our context is empty
        // AND we haven't already warmed up this conversation
        return !conversation.messages.isEmpty && (state.currentTokenCount == 0 || state.tokenBuffer.isEmpty)
    }

    // MARK: - Public API

    /// Begin a new generation and immediately obtain a token stream
    /// If no conversation is active, starts a new one
    @discardableResult
    public func start(
        prompt: String,
        params: GenerationParams,
        conversationId: ConversationID?
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

        // Start generation task within the same actor context
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
        id: GenerationID,
        prompt: String,
        params: GenerationParams,
        conversationId: ConversationID,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async {
        do {
            // Ensure context is initialized (only once per instance)
            try ensureContextInitialized()

            // Configure context for this generation (only if not already warmed up)
            let state = getOrCreateConversationState(for: conversationId)
            if state.currentTokenCount == 0 {
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
                continuation.finish(throwing: GenerationError.contextPreparationFailed(conversationId: conversationId))
                return
            }

            // Recreate sampler with updated context
            let sampler = createSampler(with: params, llamaContext: context!)
            sampler.reset()

            // Generate tokens
            var generatedTokens: [SLlamaToken] = []
            let currentState = getOrCreateConversationState(for: conversationId)
            let maxNewTokens = max(0, Int32(params.maxTokens) - currentState.currentTokenCount)

            // If we already have more tokens than requested, we can still generate a minimum number
            let minNewTokens = maxNewTokens > 0 ? maxNewTokens : 1

            // Update conversation state for generation
            updateConversationState(for: conversationId) { state in
                state.shouldContinuePredicting = true
                state.lastUsedParams = params
                state.isActive = true
            }

            while getOrCreateConversationState(for: conversationId).shouldContinuePredicting,
                  generatedTokens.count < minNewTokens
            {
                let token = predictNextToken(sampler: sampler, llamaContext: context!, conversationId: conversationId)

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
                        .tokenizationFailed(conversationId: conversationId)
                    case _ where error.localizedDescription.contains("context"):
                        .contextPreparationFailed(conversationId: conversationId)
                    default:
                        .generationFailed(generationId: id)
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
        guard !promptTokens.isEmpty else {
            return false
        }

        // Check if conversation exists
        if let conversation = conversations[conversationId] {
            // Existing conversation - continue from history
            let historyTokens = conversation.messages.flatMap(\.tokens)
            let totalTokens = historyTokens.count + promptTokens.count

            // Robust fallback: truncate oldest messages if context is too long
            if totalTokens >= maxContextSize {
                let allowed = max(0, Int(maxContextSize) - promptTokens.count)
                let trimmedHistory = Array(historyTokens.suffix(allowed))
                let adjustedTotalTokens = trimmedHistory.count + promptTokens.count

                // Validate that our truncation worked correctly
                // Note: This validation is tested indirectly in contextTruncationValidationTest
                // which verifies that context preparation fails when tokens exceed maxContextSize
                guard adjustedTotalTokens <= maxContextSize else {
                    return false
                }

                // Continue with trimmed history
                let needsContextClear = trimmedHistory.isEmpty ||
                    !getOrCreateConversationState(for: conversationId).isWarmedUp ||
                    (getOrCreateConversationState(for: conversationId).currentTokenCount == 0 && !trimmedHistory
                        .isEmpty
                    ) ||
                    (getOrCreateConversationState(for: conversationId)
                        .currentTokenCount > 0 && getOrCreateConversationState(for: conversationId).tokenBuffer.isEmpty
                    ) ||
                    (getOrCreateConversationState(for: conversationId)
                        .currentTokenCount > 0 && getOrCreateConversationState(for: conversationId).tokenBuffer
                        .count != getOrCreateConversationState(for: conversationId).currentTokenCount
                    )

                if needsContextClear {
                    // Clear the KV cache completely for a fresh start
                    llamaContext.clearMemory(data: true)

                    updateConversationState(for: conversationId) { state in
                        state.currentTokenCount = 0
                        state.tokenBuffer.removeAll()
                    }

                    clearBatch()

                    // If we have trimmed history, we need to warm up the context first
                    if !trimmedHistory.isEmpty {
                        // Warm up with trimmed historical tokens
                        for (i, token) in trimmedHistory.enumerated() {
                            addToBatch(token: token, position: Int32(i), isLogit: false)
                        }

                        do {
                            try llamaContext.core().decode(batch!)
                        } catch {
                            return false
                        }

                        // Update state with trimmed historical tokens
                        updateConversationState(for: conversationId) { state in
                            state.currentTokenCount = Int32(trimmedHistory.count)
                            state.tokenBuffer = trimmedHistory
                        }
                    }
                }

                // Now add the new prompt tokens
                clearBatch()

                let startPosition = getOrCreateConversationState(for: conversationId).currentTokenCount
                for (i, token) in promptTokens.enumerated() {
                    let position = startPosition + Int32(i)
                    addToBatch(token: token, position: position, isLogit: i == promptTokens.count - 1)
                }

                do {
                    try llamaContext.core().decode(batch!)
                } catch {
                    return false
                }

                // Update token count
                updateConversationState(for: conversationId) { state in
                    state.currentTokenCount = startPosition + Int32(promptTokens.count)
                    state.shouldContinuePredicting = true
                }

                return true
            }

            // Get current conversation state to check if we need to clear context
            let currentState = getOrCreateConversationState(for: conversationId)

            // Only clear context if:
            // 1. This is a new conversation (no history)
            // 2. This conversation hasn't been warmed up yet
            // 3. We're switching from a different conversation that wasn't properly warmed up
            // 4. The conversation state is inconsistent (has history but no token count)
            // 5. We're continuing a conversation that was loaded from JSON (needs fresh context)
            let needsContextClear = historyTokens.isEmpty ||
                !currentState.isWarmedUp ||
                (currentState.currentTokenCount == 0 && !historyTokens.isEmpty) ||
                (currentState.currentTokenCount > 0 && currentState.tokenBuffer.isEmpty) ||
                (currentState.currentTokenCount > 0 && currentState.tokenBuffer.count != currentState.currentTokenCount)

            if needsContextClear {
                // Clear the KV cache completely for a fresh start
                llamaContext.clearMemory(data: true)

                updateConversationState(for: conversationId) { state in
                    state.currentTokenCount = 0
                    state.tokenBuffer.removeAll()
                }

                clearBatch()

                // If we have history, we need to warm up the context first
                if !historyTokens.isEmpty {
                    // Warm up with historical tokens
                    for (i, token) in historyTokens.enumerated() {
                        addToBatch(token: token, position: Int32(i), isLogit: false)
                    }

                    do {
                        try llamaContext.core().decode(batch!)
                    } catch {
                        return false
                    }

                    // Update state with historical tokens
                    updateConversationState(for: conversationId) { state in
                        state.currentTokenCount = Int32(historyTokens.count)
                        state.tokenBuffer = historyTokens
                    }
                }
            }

            // Now add the new prompt tokens
            clearBatch()

            let startPosition = currentState.currentTokenCount
            for (i, token) in promptTokens.enumerated() {
                let position = startPosition + Int32(i)
                addToBatch(token: token, position: position, isLogit: i == promptTokens.count - 1)
            }

            do {
                try llamaContext.core().decode(batch!)
            } catch {
                return false
            }

            // Update token count
            updateConversationState(for: conversationId) { state in
                state.currentTokenCount = startPosition + Int32(promptTokens.count)
                state.shouldContinuePredicting = true
            }

            return true
        } else {
            // New conversation - start fresh
            // Clear the KV cache completely for a brand-new conversation
            llamaContext.clearMemory(data: true)

            // Initialize conversation state
            updateConversationState(for: conversationId) { state in
                state.currentTokenCount = 0
                state.tokenBuffer.removeAll()
            }

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

            updateConversationState(for: conversationId) { state in
                state.currentTokenCount = Int32(promptTokens.count)
                state.shouldContinuePredicting = true
            }

            return true
        }
    }

    // MARK: - Context Preparation - for new conversations

    private func prepareContext(for promptTokens: [SLlamaToken], llamaContext _: SLlamaContext) -> Bool {
        guard !promptTokens.isEmpty else { return false }

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

    // MARK: - Token Prediction

    private func predictNextToken(
        sampler: SLlamaSampler,
        llamaContext _: SLlamaContext,
        conversationId: ConversationID
    )
        -> SLlamaToken
    {
        let state = getOrCreateConversationState(for: conversationId)

        guard state.shouldContinuePredicting, state.currentTokenCount < maxContextSize else {
            return vocab.eosToken
        }

        // Sample next token
        guard let token = sampler.sample() else {
            updateConversationState(for: conversationId) { state in
                state.shouldContinuePredicting = false
            }
            return vocab.eosToken
        }

        // Accept the token
        sampler.accept(token)

        updateConversationState(for: conversationId) { state in
            state.tokenBuffer.append(token)
        }

        // Check for stop conditions
        if token == vocab.eosToken {
            updateConversationState(for: conversationId) { state in
                state.shouldContinuePredicting = false
            }
            return vocab.eosToken
        }

        // Process the token for next generation
        clearBatch()
        addToBatch(token: token, position: state.currentTokenCount)

        do {
            try context!.core().decode(batch!)
        } catch {
            updateConversationState(for: conversationId) { state in
                state.shouldContinuePredicting = false
            }
            return vocab.eosToken
        }

        updateConversationState(for: conversationId) { state in
            state.currentTokenCount += 1
        }

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
