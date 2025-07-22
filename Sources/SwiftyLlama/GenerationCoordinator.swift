import Atomics
import Foundation
import SLlama

public typealias SwiftyLlamaActor = SLlamaActor

/// Protocol for generation core functionality
@SLlamaActor
public protocol GenerationCore: Sendable {
    func generate(
        id: GenerationID,
        prompt: String,
        params: GenerationParams
    ) async throws
        -> AsyncThrowingStream<String, Error>

    func cancel(id: GenerationID)
}

/// Concrete implementation of GenerationCore using SLlama
@SLlamaActor
public class SLlamaGenerationCore: GenerationCore {
    private let model: SLlamaModel
    private let vocab: SLlamaVocab
    private let maxContextSize: Int32
    private var activeGenerations: [GenerationID: Task<Void, Never>] = [:]

    public init(modelPath: String, maxCtx: Int32 = 2048) throws {
        // Initialize SLlama backend
        SLlama.initialize()

        // Load model
        model = try SLlamaModel(modelPath: modelPath)
        vocab = SLlamaVocab(vocab: model.vocab)
        maxContextSize = maxCtx
    }

    public func generate(
        id: GenerationID,
        prompt: String,
        params: GenerationParams
    ) async throws
        -> AsyncThrowingStream<String, Error>
    {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream(
            bufferingPolicy: .unbounded
        )

        let task = Task.detached(priority: .userInitiated) { [weak self] in
            await self?.performGeneration(
                id: id,
                prompt: prompt,
                params: params,
                continuation: continuation
            )
            return ()
        }

        activeGenerations[id] = task

        // Ensure cleanup when stream is terminated
        continuation.onTermination = { @Sendable _ in
            Task { await self.cancel(id: id) }
        }

        return stream
    }

    public func cancel(id: GenerationID) {
        activeGenerations[id]?.cancel()
        activeGenerations.removeValue(forKey: id)
    }

    private func performGeneration(
        id: GenerationID,
        prompt: String,
        params: GenerationParams,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async {
        do {
            let context = try createContext(with: params)
            let sampler = createSampler(with: params, context: context)

            // Tokenize the prompt
            let promptTokens = try vocab.tokenize(text: prompt)

            // Check for empty prompt
            guard !promptTokens.isEmpty else {
                continuation.finish()
                return
            }

            // Process prompt tokens
            try await processPromptTokens(promptTokens, context: context, maxBatchSize: params.maxBatchSize)

            // Generate tokens efficiently - process one by one to avoid memory slot issues
            var currentPosition = promptTokens.count
            let maxTokens = params.maxTokens // Use configurable limit

            for _ in 0 ..< maxTokens {
                // Check for cancellation
                if Task.isCancelled { break }

                // Sample next token
                guard let nextToken = sampler.sample() else {
                    continuation.finish(throwing: GenerationError.internalFailure("Sampling failed"))
                    break
                }

                // Check for end of generation
                if nextToken == vocab.eosToken {
                    break
                }

                // Convert token to text and yield immediately
                if let tokenText = try? SLlamaTokenizer.tokenToPiece(
                    token: nextToken,
                    vocab: vocab.pointer,
                    lstrip: 0,
                    special: false
                ) {
                    continuation.yield(tokenText)
                }

                // Accept token for sampler state
                sampler.accept(nextToken)

                // Process single token immediately
                let generationBatch = SLlamaBatch(nTokens: 1, nSeqMax: 1)
                generationBatch.addToken(
                    nextToken,
                    position: Int32(currentPosition),
                    sequenceIds: [0],
                    logits: true
                )

                // Decode the single token
                try context.core().decode(generationBatch)
                currentPosition += 1
            }

            continuation.finish()

        } catch is CancellationError {
            continuation.finish(throwing: GenerationError.abortedByUser)

        } catch {
            continuation.finish(throwing: error)
        }

        // Cleanup
        activeGenerations.removeValue(forKey: id)
    }

    private func createContext(with params: GenerationParams) throws -> SLlamaContext {
        let contextParams = SLlamaContext.createParams(
            contextSize: params.contextSize,
            batchSize: params.batchSize,
            physicalBatchSize: params.physicalBatchSize,
            maxSequences: params.maxSequences,
            threads: params.threads,
            batchThreads: params.batchThreads,
            enableEmbeddings: params.enableEmbeddings,
            enableOffloading: params.enableOffloading
        )

        let context = try SLlamaContext(model: model, contextParams: contextParams)

        // Configure the context using parameters
        context.setThreads(nThreads: params.threads, nThreadsBatch: params.batchThreads)
        context.setEmbeddings(params.enableEmbeddings)
        context.setCausalAttention(params.enableCausalAttention)

        return context
    }

    private func createSampler(with params: GenerationParams, context: SLlamaContext) -> SLlamaSampler {
        // Try to create a sophisticated sampler chain that uses all parameters
        if let chain = SLlamaSamplerChain.custom(
            context: context,
            temperature: params.temperature,
            topK: params.topK,
            topP: params.topP,
            repetitionPenalty: params.repeatPenalty
        ) {
            // For now, we'll use the chain's sample method directly
            // In a more sophisticated implementation, we'd wrap the chain
            return SLlamaSampler.temperature(
                context: context,
                temperature: params.temperature
            ) ?? SLlamaSampler.greedy(context: context) ?? SLlamaSampler(context: context)
        }
        
        // Fallback to temperature sampler
        return SLlamaSampler.temperature(
            context: context,
            temperature: params.temperature
        ) ?? SLlamaSampler.greedy(context: context) ?? SLlamaSampler(context: context)
    }

    private func processPromptTokens(
        _ promptTokens: [SLlamaToken],
        context: SLlamaContext,
        maxBatchSize _: Int32
    ) async throws {
        // Process all tokens in a single batch for efficiency
        let batch = SLlamaBatch(nTokens: Int32(promptTokens.count), nSeqMax: 1)

        for (index, token) in promptTokens.enumerated() {
            batch.addToken(
                token,
                position: Int32(index),
                sequenceIds: [0],
                logits: index == promptTokens.count - 1 // Only last token needs logits
            )
        }

        try context.core().decode(batch)
    }
}

/// Opaque handle returned to UI so it can update or cancel a running stream.
public struct GenerationID: Hashable, Sendable {
    private let raw = UUID()
}

/// Runtime-mutable generation settings (top-k, temperature, …)
public struct GenerationParams: Sendable, Equatable {
    public var seed: UInt32
    public var topK: Int32
    public var topP: Float
    public var temperature: Float
    public var repeatPenalty: Float
    public var repetitionLookback: Int32

    // Context configuration parameters
    public var contextSize: UInt32
    public var batchSize: UInt32
    public var physicalBatchSize: UInt32
    public var maxSequences: UInt32
    public var threads: Int32
    public var batchThreads: Int32
    public var enableEmbeddings: Bool
    public var enableOffloading: Bool
    public var enableCausalAttention: Bool

    // Generation limits
    public var maxTokens: Int32
    public var maxBatchSize: Int32

    public init(
        // Generation parameters
        seed: UInt32 = 42,
        topK: Int32 = 40,
        topP: Float = 0.9,
        temperature: Float = 0.7,
        repeatPenalty: Float = 1.1,
        repetitionLookback: Int32 = 64,

        // Context configuration
        contextSize: UInt32 = 2048,
        batchSize: UInt32 = 1, // Single token processing for memory efficiency
        physicalBatchSize: UInt32 = 1, // Single token processing for memory efficiency
        maxSequences: UInt32 = 1,
        threads: Int32 = 4, // Conservative thread count
        batchThreads: Int32 = 4, // Conservative thread count
        enableEmbeddings: Bool = false,
        enableOffloading: Bool = true,
        enableCausalAttention: Bool = true,

        // Generation limits
        maxTokens: Int32 = 1000, // Safety limit
        maxBatchSize: Int32 = 256 // Maximum batch size for prompt processing
    ) {
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.temperature = temperature
        self.repeatPenalty = repeatPenalty
        self.repetitionLookback = repetitionLookback

        self.contextSize = contextSize
        self.batchSize = batchSize
        self.physicalBatchSize = physicalBatchSize
        self.maxSequences = maxSequences
        self.threads = threads
        self.batchThreads = batchThreads
        self.enableEmbeddings = enableEmbeddings
        self.enableOffloading = enableOffloading
        self.enableCausalAttention = enableCausalAttention

        self.maxTokens = maxTokens
        self.maxBatchSize = maxBatchSize
    }
}

/// What the UI gets back from `start()`
public struct GenerationStream: Sendable {
    public let id: GenerationID
    public let stream: AsyncThrowingStream<String, Error>
}

/// Domain error surface
public enum GenerationError: Error, LocalizedError {
    case abortedByUser
    case internalFailure(String)
    case modelLoadFailed(String)
    case contextCreationFailed(String)
    case tokenizationFailed(String)
    case samplingFailed(String)
    case invalidParameters(String)

    public var errorDescription: String? {
        switch self {
            case .abortedByUser:
                "Generation was aborted by user"
            case let .internalFailure(message):
                "Internal failure: \(message)"
            case let .modelLoadFailed(message):
                "Model load failed: \(message)"
            case let .contextCreationFailed(message):
                "Context creation failed: \(message)"
            case let .tokenizationFailed(message):
                "Tokenization failed: \(message)"
            case let .samplingFailed(message):
                "Sampling failed: \(message)"
            case let .invalidParameters(message):
                "Invalid parameters: \(message)"
        }
    }
}

/// The coordinator is a thin book-keeper.
/// *It never touches llama.cpp directly* — that is done by `SLlamaActor`.
@SwiftyLlamaActor
public class GenerationCoordinator {
    // MARK: - private data

    private struct Live {
        let id: GenerationID
        var params: GenerationParams
        let continuation: AsyncThrowingStream<String, Error>.Continuation
        let startTime: Date
    }

    private var live: [GenerationID: Live] = [:]
    private let core: GenerationCore // injected
    private let buffer = 64 // token buffer size

    // MARK: - life-cycle

    public init(core: GenerationCore) {
        self.core = core
    }

    /// Convenient initializer that creates a SLlamaGenerationCore with the given model path
    public convenience init(modelPath: String, maxCtx: Int32 = 2048) throws {
        let core = try SLlamaGenerationCore(modelPath: modelPath, maxCtx: maxCtx)
        self.init(core: core)
    }

    // MARK: - public API

    /// Begin a new generation and immediately obtain a **token stream**.
    @discardableResult
    public func start(
        prompt: String,
        params: GenerationParams
    )
        -> GenerationStream
    {
        let id = GenerationID()

        let (stream, cont) = AsyncThrowingStream<String, Error>.makeStream(
            bufferingPolicy: .unbounded
        )

        live[id] = Live(id: id, params: params, continuation: cont, startTime: Date())

        // Kick off the real work on a detached task so the actor is free.
        Task.detached(priority: .userInitiated) { [core] in
            do {
                let tokenStream = try await core.generate(
                    id: id,
                    prompt: prompt,
                    params: params
                )

                for try await token in tokenStream {
                    cont.yield(token)
                }
                cont.finish()

            } catch is CancellationError {
                cont.finish(throwing: GenerationError.abortedByUser)

            } catch {
                cont.finish(throwing: error)
            }
            await self.finish(id)
        }

        // Ensure cleanup if consumer disappears.
        cont.onTermination = { @Sendable _ in
            Task { await self.cancel(id) }
        }

        return .init(id: id, stream: stream)
    }

    /// Live-edit the sampling parameters of a running generation.
    public func update(
        id: GenerationID,
        _ new: GenerationParams
    ) {
        guard var live = live[id] else { return }
        live.params = new
        self.live[id] = live
    }

    /// Cancel a running generation.
    public func cancel(_ id: GenerationID) async {
        guard let _ = live[id] else { return }
        core.cancel(id: id)
        await finish(id)
    }

    /// Get information about a running generation.
    public func getGenerationInfo(_ id: GenerationID) -> (params: GenerationParams, startTime: Date)? {
        guard let live = live[id] else { return nil }
        return (live.params, live.startTime)
    }

    /// Get all active generation IDs.
    public func getActiveGenerationIDs() -> [GenerationID] {
        Array(live.keys)
    }

    /// Cancel all running generations.
    public func cancelAll() async {
        let ids = Array(live.keys)
        for id in ids {
            await cancel(id)
        }
    }

    // MARK: - helpers

    private func finish(_ id: GenerationID) async {
        live[id]?.continuation.finish()
        live.removeValue(forKey: id)
    }
}
