import Atomics
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

    // Generation tracking
    private struct LiveGeneration {
        let id: GenerationID
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
        guard let context else { throw NSError(
            domain: "SwiftyCoreLlama",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Context not initialized"]
        ) }

        // Configure the context using parameters
        context.setThreads(nThreads: params.threads, nThreadsBatch: params.batchThreads)
        context.setEmbeddings(params.enableEmbeddings)
        context.setCausalAttention(params.enableCausalAttention)
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

    // MARK: - Public API

    /// Begin a new generation and immediately obtain a token stream
    @discardableResult
    public func start(
        prompt: String,
        params: GenerationParams
    ) async
        -> GenerationStream
    {
        let id = GenerationID()

        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream(
            bufferingPolicy: .unbounded
        )

        // Store generation info
        activeGenerations[id] = LiveGeneration(
            id: id,
            params: params,
            startTime: Date(),
            task: nil
        )

        // Start generation task within the same actor context - NO detached task!
        Task { @SwiftyLlamaActor in
            await self.performGeneration(
                id: id,
                prompt: prompt,
                params: params,
                continuation: continuation
            )
            return ()
        }

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
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async {
        do {
            // Ensure context is initialized (only once per instance)
            try ensureContextInitialized()

            // Configure context for this generation
            try configureContext(with: params)

            // Create sampler
            let sampler = createSampler(with: params, context: context!)

            // Tokenize prompt
            let promptTokens = try vocab.tokenize(text: prompt)

            // Prepare context (like the working example)
            guard prepareContext(for: promptTokens, context: context!) else {
                continuation.finish(throwing: NSError(
                    domain: "SwiftyCoreLlama",
                    code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to prepare context"]
                ))
                return
            }

            // Reset sampler state
            sampler.reset()

            // Generate tokens
            while shouldContinuePredicting, currentTokenCount < Int32(params.maxTokens) {
                let token = predictNextToken(sampler: sampler, context: context!)

                if token == vocab.eosToken {
                    break
                }

                let tokenText = try vocab.tokenToPiece(token: token)
                continuation.yield(tokenText)
            }

            continuation.finish()

        } catch {
            continuation.finish(throwing: error)
        }
    }

    // MARK: - Context Preparation (like the working example)

    private func prepareContext(for promptTokens: [SLlamaToken], context: SLlamaContext) -> Bool {
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
            try context.core().decode(batch!)
        } catch {
            return false
        }

        currentTokenCount = Int32(initialCount)
        shouldContinuePredicting = true
        return true
    }

    // MARK: - Token Prediction (like the working example)

    private func predictNextToken(sampler: SLlamaSampler, context: SLlamaContext) -> SLlamaToken {
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
            try context.core().decode(batch!)
        } catch {
            shouldContinuePredicting = false
            return vocab.eosToken
        }

        currentTokenCount += 1
        return token
    }

    private func createSampler(with params: GenerationParams, context: SLlamaContext) -> SLlamaSampler {
        // Create a temperature sampler with the given parameters
        let sampler = SLlamaSampler.temperature(
            context: context,
            temperature: params.temperature
        ) ?? SLlamaSampler.greedy(context: context) ?? SLlamaSampler(context: context)

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

// MARK: - Supporting Types

public struct GenerationInfo {
    public let id: GenerationID
    public let params: GenerationParams
    public let startTime: Date
    public let isActive: Bool
}

public struct ModelInfo {
    public let name: String
    public let contextSize: Int32
    public let vocabSize: Int32
}

public struct VocabInfo {
    public let size: Int32
    public let bosToken: SLlamaToken
    public let eosToken: SLlamaToken
    public let nlToken: SLlamaToken
}
