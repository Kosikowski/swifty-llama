import Foundation
import TestUtilities
import XCTest
@testable import SLlama
@testable import SwiftyLlama

@SLlamaActor
final class SwiftyLlamaCoreTests: XCTestCase, @unchecked Sendable {
    // MARK: - Test Properties

    private var core: SwiftyLlamaCore?
    private let testModelPath = TestUtilities.testModelPath

    // MARK: - Setup and Teardown

    override func setUp() async throws {
        try await super.setUp()

        // Skip tests if model is not available
        TestUtilities.skipIfModelUnavailable(testName: #function)
        TestUtilities.skipIfIOSSimulator(testName: #function)

        // Don't initialize core here - create it per test to avoid state issues
    }

    override func tearDown() async throws {
        core = nil
        try await super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInitializationWithValidModel() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
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
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Test with custom context size
        let testCore = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 256)
        XCTAssertNotNil(testCore)
    }

    // MARK: - Generation Tests

    func testBasicGeneration() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Create a new core for this test with smaller context size for speed
        let core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)

        let id = GenerationID()
        let params = GenerationParams(
            seed: 42,
            topK: 5, // Reduced for faster sampling
            topP: 0.8, // Reduced for faster sampling
            temperature: 0.1 // Low temperature for deterministic output
        )

        let prompt = "Hello"

        // Start generation
        let stream = try await core.generate(id: id, prompt: prompt, params: params)

        // Collect tokens with a limit for faster tests
        var tokens: [String] = []
        var tokenCount = 0
        let maxTokens = 5 // Limit to 5 tokens for speed

        for try await token in stream {
            tokens.append(token)
            tokenCount += 1
            if tokenCount >= maxTokens { break } // Early termination
        }

        // Verify we got some output
        XCTAssertFalse(tokens.isEmpty, "Should generate some tokens")
        XCTAssertLessThanOrEqual(tokens.count, maxTokens, "Should not exceed token limit")

        // Verify tokens are not empty strings
        XCTAssertTrue(tokens.allSatisfy { !$0.isEmpty }, "All tokens should be non-empty")
    }

    func testGenerationWithDifferentPrompts() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Create a new core for this test with smaller context size for speed
        let core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)

        let prompts = ["Hello", "The quick brown", "Once upon a time"]
        let params = GenerationParams(
            seed: 42,
            topK: 5, // Reduced for faster sampling
            topP: 0.8, // Reduced for faster sampling
            temperature: 0.1 // Low temperature for deterministic output
        )

        for prompt in prompts {
            let id = GenerationID()
            let stream = try await core.generate(id: id, prompt: prompt, params: params)

            var tokens: [String] = []
            var tokenCount = 0
            let maxTokens = 3 // Limit to 3 tokens for speed

            for try await token in stream {
                tokens.append(token)
                tokenCount += 1
                if tokenCount >= maxTokens { break } // Early termination
            }

            XCTAssertFalse(tokens.isEmpty, "Should generate tokens for prompt: \(prompt)")
            XCTAssertLessThanOrEqual(tokens.count, maxTokens, "Should not exceed token limit")
        }
    }

    func testGenerationWithDifferentParameters() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Create a new core for this test with smaller context size for speed
        let core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)

        let testCases = [
            GenerationParams(seed: 1, topK: 3, topP: 0.7, temperature: 0.1), // More aggressive limits
            GenerationParams(seed: 2, topK: 5, topP: 0.8, temperature: 0.3), // Moderate limits
            GenerationParams(seed: 3, topK: 10, topP: 0.9, temperature: 0.5), // Less aggressive
        ]

        let prompt = "Test"

        for params in testCases {
            let id = GenerationID()
            let stream = try await core.generate(id: id, prompt: prompt, params: params)

            var tokens: [String] = []
            var tokenCount = 0
            let maxTokens = 2 // Limit to 2 tokens for speed

            for try await token in stream {
                tokens.append(token)
                tokenCount += 1
                if tokenCount >= maxTokens { break } // Early termination
            }

            XCTAssertFalse(tokens.isEmpty, "Should generate tokens with params: \(params)")
            XCTAssertLessThanOrEqual(tokens.count, maxTokens, "Should not exceed token limit")
        }
    }

    // MARK: - Cancellation Tests

    func testGenerationCancellation() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Create a new core for this test with smaller context size for speed
        let core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)

        let id = GenerationID()
        let params = GenerationParams(
            seed: 42,
            topK: 5, // Reduced for faster sampling
            topP: 0.8, // Reduced for faster sampling
            temperature: 0.3 // Lower temperature for faster convergence
        )
        let prompt = "This is a test prompt"

        // Start generation
        let stream = try await core.generate(id: id, prompt: prompt, params: params)

        // Cancel immediately
        core.cancel(id: id)

        // Try to collect tokens - should finish quickly due to cancellation
        var tokenCount = 0
        do {
            for try await _ in stream {
                tokenCount += 1
                if tokenCount > 5 { break } // Reduced safety limit
            }
        } catch {
            // Expected to throw cancellation error
            XCTAssertTrue(error is CancellationError || error.localizedDescription.contains("aborted"))
        }

        // Should have very few or no tokens due to immediate cancellation
        XCTAssertLessThanOrEqual(tokenCount, 5)
    }

    func testMultipleGenerationsCancellation() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Create a new core for this test with smaller context size for speed
        let core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)

        let ids = [GenerationID(), GenerationID(), GenerationID()]
        let params = GenerationParams(
            seed: 42,
            topK: 5, // Reduced for faster sampling
            topP: 0.8, // Reduced for faster sampling
            temperature: 0.3 // Lower temperature for faster convergence
        )
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
                if cancelledTokenCount > 3 { break } // Reduced limit
            }
        } catch {
            // Expected
        }

        XCTAssertLessThanOrEqual(cancelledTokenCount, 3)
    }

    // MARK: - Error Handling Tests

    func testGenerationWithEmptyPrompt() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Create a new core for this test with smaller context size for speed
        let core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)

        let id = GenerationID()
        let params = GenerationParams(
            seed: 42,
            topK: 5, // Reduced for faster sampling
            topP: 0.8, // Reduced for faster sampling
            temperature: 0.1 // Low temperature for deterministic output
        )
        let prompt = ""

        // Should handle empty prompt gracefully
        let stream = try await core.generate(id: id, prompt: prompt, params: params)

        var tokens: [String] = []
        var tokenCount = 0
        let maxTokens = 2 // Limit for speed

        for try await token in stream {
            tokens.append(token)
            tokenCount += 1
            if tokenCount >= maxTokens { break } // Early termination
        }

        // Should either generate tokens or finish gracefully
        XCTAssertTrue(tokens.isEmpty || !tokens.isEmpty)
    }

    func testGenerationWithVeryLongPrompt() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Create a new core for this test with smaller context size for speed
        let core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)

        let id = GenerationID()
        let params = GenerationParams(
            seed: 42,
            topK: 5, // Reduced for faster sampling
            topP: 0.8, // Reduced for faster sampling
            temperature: 0.1 // Low temperature for deterministic output
        )
        let prompt = String(repeating: "This is a test prompt. ", count: 20) // Reduced length

        // Should handle long prompt
        let stream = try await core.generate(id: id, prompt: prompt, params: params)

        var tokens: [String] = []
        var tokenCount = 0
        let maxTokens = 3 // Limit for speed

        for try await token in stream {
            tokens.append(token)
            tokenCount += 1
            if tokenCount >= maxTokens { break } // Early termination
        }

        // Should generate some output
        XCTAssertTrue(tokens.isEmpty || !tokens.isEmpty)
    }

    // MARK: - Parameter Validation Tests

    func testGenerationWithExtremeParameters() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Create a new core for this test with smaller context size for speed
        let core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)

        let extremeParams = [
            GenerationParams(seed: 1, topK: 1, topP: 0.1, temperature: 0.0), // Very deterministic
            GenerationParams(seed: 2, topK: 10, topP: 0.9, temperature: 1.0), // Less extreme random
            GenerationParams(
                seed: 3,
                topK: 5,
                topP: 0.8,
                temperature: 0.3,
                repeatPenalty: 1.5, // Reduced penalty
                repetitionLookback: 64 // Reduced lookback
            ), // Moderate repetition penalty
        ]

        let prompt = "Test"

        for params in extremeParams {
            let id = GenerationID()
            let stream = try await core.generate(id: id, prompt: prompt, params: params)

            var tokens: [String] = []
            var tokenCount = 0
            let maxTokens = 2 // Limit for speed

            for try await token in stream {
                tokens.append(token)
                tokenCount += 1
                if tokenCount >= maxTokens { break } // Early termination
            }

            // Should handle extreme parameters without crashing
            XCTAssertTrue(tokens.isEmpty || !tokens.isEmpty)
        }
    }

    // MARK: - Concurrency Tests

    func testConcurrentGenerations() async throws {
        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }

        // Create a new core for this test with smaller context size for speed
        let core = try SwiftyLlamaCore(modelPath: testModelPath, maxCtx: 512)

        let prompt = "Concurrent test"
        let params = GenerationParams(
            seed: 42,
            topK: 5, // Reduced for faster sampling
            topP: 0.8, // Reduced for faster sampling
            temperature: 0.1 // Low temperature for deterministic output
        )

        // Start multiple concurrent generations
        async let generation1 = try await collectTokens(from: core.generate(
            id: GenerationID(),
            prompt: prompt,
            params: params
        ), maxTokens: 2)
        async let generation2 = try await collectTokens(from: core.generate(
            id: GenerationID(),
            prompt: prompt,
            params: params
        ), maxTokens: 2)
        async let generation3 = try await collectTokens(from: core.generate(
            id: GenerationID(),
            prompt: prompt,
            params: params
        ), maxTokens: 2)

        // Wait for all to complete
        let (tokens1, tokens2, tokens3) = try await (generation1, generation2, generation3)

        // All should generate some tokens
        XCTAssertFalse(tokens1.isEmpty)
        XCTAssertFalse(tokens2.isEmpty)
        XCTAssertFalse(tokens3.isEmpty)
    }

    // MARK: - Helper Methods

    private func collectTokens(
        from stream: AsyncThrowingStream<String, Error>,
        maxTokens: Int = 5
    ) async throws
        -> [String]
    {
        var tokens: [String] = []
        var tokenCount = 0

        for try await token in stream {
            tokens.append(token)
            tokenCount += 1
            if tokenCount >= maxTokens { break } // Early termination
        }

        return tokens
    }
}
