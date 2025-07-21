import Foundation
import Atomics
import SLlama

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
    
    public init(
        seed: UInt32 = 42,
        topK: Int32 = 40,
        topP: Float = 0.9,
        temperature: Float = 0.7,
        repeatPenalty: Float = 1.1,
        repetitionLookback: Int32 = 64
    ) {
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.temperature = temperature
        self.repeatPenalty = repeatPenalty
        self.repetitionLookback = repetitionLookback
    }
}

/// What the UI gets back from `start()`
public struct GenerationStream {
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
            return "Generation was aborted by user"
        case .internalFailure(let message):
            return "Internal failure: \(message)"
        case .modelLoadFailed(let message):
            return "Model load failed: \(message)"
        case .contextCreationFailed(let message):
            return "Context creation failed: \(message)"
        case .tokenizationFailed(let message):
            return "Tokenization failed: \(message)"
        case .samplingFailed(let message):
            return "Sampling failed: \(message)"
        case .invalidParameters(let message):
            return "Invalid parameters: \(message)"
        }
    }
}

/// The coordinator is a thin book-keeper.  
/// *It never touches llama.cpp directly* — that is done by `LlamaCoreActor`.
public actor GenerationCoordinator {

    // MARK: - private data

    private struct Live {
        let id: GenerationID
        var params: GenerationParams
        let continuation: AsyncThrowingStream<String, Error>.Continuation
        let startTime: Date
    }

    private var live: [GenerationID: Live] = [:]
    private let core: LlamaCoreActor            // injected
    private let buffer = 64                     // token buffer size

    // MARK: - life-cycle

    public init(core: LlamaCoreActor) {
        self.core = core
    }

    // MARK: - public API

    /// Begin a new generation and immediately obtain a **token stream**.
    @discardableResult
    public func start(prompt: String,
                      params: GenerationParams) -> GenerationStream {

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
    public func update(id: GenerationID,
                       _ new: GenerationParams) {
        guard var live = live[id] else { return }
        live.params = new
        self.live[id] = live
    }

    /// Cancel a running generation.
    public func cancel(_ id: GenerationID) async {
        guard let _ = live[id] else { return }
        await core.cancel(id: id)
        finish(id)
    }
    
    /// Get information about a running generation.
    public func getGenerationInfo(_ id: GenerationID) -> (params: GenerationParams, startTime: Date)? {
        guard let live = live[id] else { return nil }
        return (live.params, live.startTime)
    }
    
    /// Get all active generation IDs.
    public func getActiveGenerationIDs() -> [GenerationID] {
        return Array(live.keys)
    }
    
    /// Cancel all running generations.
    public func cancelAll() async {
        let ids = Array(live.keys)
        for id in ids {
            await cancel(id)
        }
    }

    // MARK: - helpers

    private func finish(_ id: GenerationID) {
        live[id]?.continuation.finish()
        live.removeValue(forKey: id)
    }
}
