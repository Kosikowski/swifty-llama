import Foundation
import SLlama

/// A demonstration of the SwiftyCoreLlama concept that solves async stream context issues
/// This shows how combining GenerationCoordinator and SLlamaGenerationCore into a single actor
/// solves the async stream context problems without using the real model
@SLlamaActor
public class SwiftyCoreLlamaDemo {
    // MARK: - Private Properties

    private var activeGenerations: [GenerationID: LiveGeneration] = [:]

    // Generation tracking
    private struct LiveGeneration {
        let id: GenerationID
        var params: GenerationParams
        let startTime: Date
        var task: Task<Void, Never>?
    }

    // MARK: - Initialization

    public init() {
        // Simple initialization without model loading
    }

    // MARK: - Public API

    /// Begin a new generation and immediately obtain a token stream
    @discardableResult
    public func start(
        prompt: String,
        params: GenerationParams
    )
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
            task: nil as Task<Void, Never>?
        )

        // Start generation task within the same actor context
        let task = Task.detached(priority: .userInitiated) { [weak self] in
            await self?.performMockGeneration(
                id: id,
                prompt: prompt,
                params: params,
                continuation: continuation
            )
            return ()
        }

        // Update the task reference
        if var generation = activeGenerations[id] {
            generation.task = task
            activeGenerations[id] = generation
        }

        // Ensure cleanup when stream is terminated
        continuation.onTermination = { @Sendable _ in
            Task { await self.cancel(id) }
        }

        return GenerationStream(id: id, stream: stream)
    }

    /// Live-edit the sampling parameters of a running generation
    public func update(id: GenerationID, _ newParams: GenerationParams) {
        guard var generation = activeGenerations[id] else { return }
        generation.params = newParams
        activeGenerations[id] = generation
    }

    /// Cancel a running generation
    public func cancel(_ id: GenerationID) async {
        guard let generation = activeGenerations[id] else { return }

        // Cancel the task
        generation.task?.cancel()

        // Remove from active generations
        activeGenerations.removeValue(forKey: id)
    }

    /// Get information about a running generation
    public func getGenerationInfo(_ id: GenerationID) -> (params: GenerationParams, startTime: Date)? {
        guard let generation = activeGenerations[id] else { return nil }
        return (generation.params, generation.startTime)
    }

    /// Get all active generation IDs
    public func getActiveGenerationIDs() -> [GenerationID] {
        Array(activeGenerations.keys)
    }

    /// Cancel all running generations
    public func cancelAll() async {
        let ids = Array(activeGenerations.keys)
        for id in ids {
            await cancel(id)
        }
    }

    // MARK: - Private Mock Generation Logic

    private func performMockGeneration(
        id: GenerationID,
        prompt: String,
        params: GenerationParams,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async {
        do {
            // Simulate some processing time
            try await Task.sleep(nanoseconds: 100_000_000) // 100ms

            // Check for cancellation
            if Task.isCancelled {
                continuation.finish(throwing: GenerationError.abortedByUser)
                return
            }

            // Generate mock tokens based on the prompt
            let mockTokens = generateMockTokens(for: prompt, params: params)

            for token in mockTokens {
                // Check for cancellation between tokens
                if Task.isCancelled {
                    continuation.finish(throwing: GenerationError.abortedByUser)
                    return
                }

                continuation.yield(token)

                // Small delay between tokens
                try await Task.sleep(nanoseconds: 50_000_000) // 50ms
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

    private func generateMockTokens(for prompt: String, params: GenerationParams) -> [String] {
        // Generate mock tokens based on the prompt
        let words = prompt.components(separatedBy: " ")
        var tokens: [String] = []

        for (index, word) in words.enumerated() {
            if index < params.maxTokens {
                tokens.append(word)
                if index < words.count - 1 {
                    tokens.append(" ")
                }
            }
        }

        // Add some mock completion
        if tokens.count < params.maxTokens {
            tokens.append(".")
        }

        return tokens
    }
}

// MARK: - Convenience Extensions

public extension SwiftyCoreLlamaDemo {
    /// Get demo information
    var demoInfo: (name: String, description: String) {
        (name: "SwiftyCoreLlamaDemo", description: "Demonstrates async stream context solution")
    }
}
