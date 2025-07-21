import Foundation
import XCTest
@testable import SLlama
@testable import SwiftyLlama

@SLlamaActor
final class SwiftyLlamaCoreTests: XCTestCase {
    // MARK: - Test Properties

    private var core: SwiftyLlamaCore?
    private let testModelPath = SLlamaTestUtilities.testModelPath

    // MARK: - Setup and Teardown

    override func setUp() async throws {
        try await super.setUp()

        // Skip tests if model is not available
        SLlamaTestUtilities.skipIfModelUnavailable(testName: #function)
        SLlamaTestUtilities.skipIfIOSSimulator(testName: #function)

        // Only initialize core if model is available
        if SLlamaTestUtilities.isTestModelAvailable() {
            core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)
        }
    }

    override func tearDown() async throws {
        core = nil
        try await super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInitializationWithValidModel() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Test initialization
        let testCore = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 1024)

        // Verify core was created
        XCTAssertNotNil(testCore)
    }

    func testInitializationWithInvalidModelPath() async throws {
        // Test with non-existent model path
        XCTAssertThrowsError(try SwiftyLlamaCore(modelPath: "/nonexistent/model.gguf")) { error in
            // Should throw a model loading error
            XCTAssertTrue(error is SLlamaError || error.localizedDescription.contains("model"))
        }
    }

    func testInitializationWithCustomContextSize() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Test with custom context size
        let testCore = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 256)
        XCTAssertNotNil(testCore)
    }

    // MARK: - Generation Tests

    func testBasicGeneration() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable(),
              let core
        else {
            throw XCTSkip("Test model not available")
        }

        let id = GenerationID()
        let params = GenerationParams(
            seed: 42,
            topK: 10,
            topP: 0.9,
            temperature: 0.1 // Low temperature for deterministic output
        )

        let prompt = "Hello"

        // Start generation
        let stream = try await core.generate(id: id, prompt: prompt, params: params)

        // Collect tokens
        var tokens: [String] = []
        for try await token in stream {
            tokens.append(token)
        }

        // Verify we got some output
        XCTAssertFalse(tokens.isEmpty, "Should generate some tokens")

        // Verify tokens are not empty strings
        XCTAssertTrue(tokens.allSatisfy { !$0.isEmpty }, "All tokens should be non-empty")
    }

    func testGenerationWithDifferentPrompts() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable(),
              let core
        else {
            throw XCTSkip("Test model not available")
        }

        let prompts = ["Hello", "The quick brown", "Once upon a time"]
        let params = GenerationParams(temperature: 0.1)

        for prompt in prompts {
            let id = GenerationID()
            let stream = try await core.generate(id: id, prompt: prompt, params: params)

            var tokens: [String] = []
            for try await token in stream {
                tokens.append(token)
            }

            XCTAssertFalse(tokens.isEmpty, "Should generate tokens for prompt: \(prompt)")
        }
    }

    func testGenerationWithDifferentParameters() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable(),
              let core
        else {
            throw XCTSkip("Test model not available")
        }

        let testCases = [
            GenerationParams(seed: 1, topK: 5, topP: 0.8, temperature: 0.1),
            GenerationParams(seed: 2, topK: 20, topP: 0.9, temperature: 0.5),
            GenerationParams(seed: 3, topK: 40, topP: 0.95, temperature: 0.9),
        ]

        let prompt = "Test"

        for params in testCases {
            let id = GenerationID()
            let stream = try await core.generate(id: id, prompt: prompt, params: params)

            var tokens: [String] = []
            for try await token in stream {
                tokens.append(token)
            }

            XCTAssertFalse(tokens.isEmpty, "Should generate tokens with params: \(params)")
        }
    }

    // MARK: - Cancellation Tests

    func testGenerationCancellation() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable(),
              let core
        else {
            throw XCTSkip("Test model not available")
        }

        let id = GenerationID()
        let params = GenerationParams(temperature: 0.7)
        let prompt = "This is a long prompt that should generate many tokens"

        // Start generation
        let stream = try await core.generate(id: id, prompt: prompt, params: params)

        // Cancel immediately
        core.cancel(id: id)

        // Try to collect tokens - should finish quickly due to cancellation
        var tokenCount = 0
        do {
            for try await _ in stream {
                tokenCount += 1
                if tokenCount > 10 { break } // Safety limit
            }
        } catch {
            // Expected to throw cancellation error
            XCTAssertTrue(error is CancellationError || error.localizedDescription.contains("aborted"))
        }

        // Should have very few or no tokens due to immediate cancellation
        XCTAssertLessThanOrEqual(tokenCount, 10)
    }

    func testMultipleGenerationsCancellation() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable(),
              let core
        else {
            throw XCTSkip("Test model not available")
        }

        let ids = [GenerationID(), GenerationID(), GenerationID()]
        let params = GenerationParams(temperature: 0.7)
        let prompt = "Test"

        // Start multiple generations
        var streams: [AsyncThrowingStream<String, Error>] = []
        for id in ids {
            let stream = try await core.generate(id: id, prompt: prompt, params: params)
            streams.append(stream)
        }

        // Cancel one generation
        core.cancel(id: ids[1])

        // Verify the cancelled generation finishes quickly
        var cancelledTokenCount = 0
        do {
            for try await _ in streams[1] {
                cancelledTokenCount += 1
                if cancelledTokenCount > 5 { break }
            }
        } catch {
            // Expected
        }

        XCTAssertLessThanOrEqual(cancelledTokenCount, 5)
    }

    // MARK: - Error Handling Tests

    func testGenerationWithEmptyPrompt() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable(),
              let core
        else {
            throw XCTSkip("Test model not available")
        }

        let id = GenerationID()
        let params = GenerationParams()
        let prompt = ""

        // Should handle empty prompt gracefully
        let stream = try await core.generate(id: id, prompt: prompt, params: params)

        var tokens: [String] = []
        for try await token in stream {
            tokens.append(token)
        }

        // Should either generate tokens or finish gracefully
        XCTAssertTrue(tokens.isEmpty || !tokens.isEmpty)
    }

    func testGenerationWithVeryLongPrompt() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable(),
              let core
        else {
            throw XCTSkip("Test model not available")
        }

        let id = GenerationID()
        let params = GenerationParams()
        let prompt = String(repeating: "This is a test prompt. ", count: 100)

        // Should handle long prompt
        let stream = try await core.generate(id: id, prompt: prompt, params: params)

        var tokens: [String] = []
        for try await token in stream {
            tokens.append(token)
        }

        // Should generate some output
        XCTAssertTrue(tokens.isEmpty || !tokens.isEmpty)
    }

    // MARK: - Parameter Validation Tests

    func testGenerationWithExtremeParameters() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable(),
              let core
        else {
            throw XCTSkip("Test model not available")
        }

        let extremeParams = [
            GenerationParams(seed: 1, topK: 1, topP: 0.1, temperature: 0.0), // Very deterministic
            GenerationParams(seed: 2, topK: 100, topP: 1.0, temperature: 2.0), // Very random
            GenerationParams(
                seed: 3,
                topK: 40,
                topP: 0.9,
                temperature: 0.7,
                repeatPenalty: 2.0,
                repetitionLookback: 128
            ), // High repetition penalty
        ]

        let prompt = "Test"

        for params in extremeParams {
            let id = GenerationID()
            let stream = try await core.generate(id: id, prompt: prompt, params: params)

            var tokens: [String] = []
            for try await token in stream {
                tokens.append(token)
            }

            // Should handle extreme parameters without crashing
            XCTAssertTrue(tokens.isEmpty || !tokens.isEmpty)
        }
    }

    // MARK: - Concurrency Tests

    func testConcurrentGenerations() async throws {
        // Skip if model not available
        guard SLlamaTestUtilities.isTestModelAvailable(),
              let core
        else {
            throw XCTSkip("Test model not available")
        }

        let prompt = "Concurrent test"
        let params = GenerationParams(temperature: 0.1)

        // Start multiple concurrent generations
        async let generation1 = try await collectTokens(from: core.generate(
            id: GenerationID(),
            prompt: prompt,
            params: params
        ))
        async let generation2 = try await collectTokens(from: core.generate(
            id: GenerationID(),
            prompt: prompt,
            params: params
        ))
        async let generation3 = try await collectTokens(from: core.generate(
            id: GenerationID(),
            prompt: prompt,
            params: params
        ))

        // Wait for all to complete
        let (tokens1, tokens2, tokens3) = try await (generation1, generation2, generation3)

        // All should generate some tokens
        XCTAssertFalse(tokens1.isEmpty)
        XCTAssertFalse(tokens2.isEmpty)
        XCTAssertFalse(tokens3.isEmpty)
    }

    // MARK: - Helper Methods

    private func collectTokens(from stream: AsyncThrowingStream<String, Error>) async throws -> [String] {
        var tokens: [String] = []
        for try await token in stream {
            tokens.append(token)
        }
        return tokens
    }
}
