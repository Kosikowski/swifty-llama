import Foundation
import SLlama
import XCTest
@testable import SwiftyLlama

@SwiftyLlamaActor
final class SwiftyCoreLlamaDemoTests: XCTestCase, @unchecked Sendable {
    // MARK: - Test Properties

    private var demo: SwiftyCoreLlamaDemo?

    // MARK: - Setup and Teardown

    override func setUp() async throws {
        try await super.setUp()

        // Create SwiftyCoreLlamaDemo (no model loading required)
        demo = SwiftyCoreLlamaDemo()
    }

    override func tearDown() async throws {
        if let demo {
            await demo.cancelAll()
        }
        demo = nil
        try await super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInitialization() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        XCTAssertNotNil(demo)

        // Test demo info
        let demoInfo = demo.demoInfo
        XCTAssertEqual(demoInfo.name, "SwiftyCoreLlamaDemo")
        XCTAssertFalse(demoInfo.description.isEmpty)
    }

    // MARK: - Basic Functionality Tests

    func testStartGeneration() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        let prompt = "Hello world"
        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start generation
        let stream = demo.start(prompt: prompt, params: params)

        // Verify stream was created
        XCTAssertNotNil(stream)
        XCTAssertNotNil(stream.id)
        XCTAssertNotNil(stream.stream)

        // Verify generation info exists
        let info = demo.getGenerationInfo(stream.id)
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.params, params)

        // Cancel to clean up
        await demo.cancel(stream.id)
    }

    func testMultipleGenerations() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        let prompts = ["First", "Second", "Third"]
        let params = GenerationParams(temperature: 0.5, maxTokens: 3)

        var streams: [GenerationStream] = []

        // Start multiple generations
        for prompt in prompts {
            let stream = demo.start(prompt: prompt, params: params)
            streams.append(stream)
        }

        // Verify all generations were started
        XCTAssertEqual(streams.count, 3)
        XCTAssertEqual(demo.getActiveGenerationIDs().count, 3)

        // Verify all have unique IDs
        let ids = streams.map(\.id)
        XCTAssertEqual(Set(ids).count, 3, "All generation IDs should be unique")

        // Cancel all to clean up
        await demo.cancelAll()
    }

    // MARK: - Parameter Update Tests

    func testUpdateGenerationParameters() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        let initialParams = GenerationParams(seed: 1, topK: 20, temperature: 0.5, maxTokens: 3)
        let updatedParams = GenerationParams(seed: 2, topK: 40, temperature: 0.8, maxTokens: 3)

        // Start generation
        let stream = demo.start(prompt: "Test", params: initialParams)

        // Update parameters
        demo.update(id: stream.id, updatedParams)

        // Verify parameters were updated
        let info = demo.getGenerationInfo(stream.id)
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.params, updatedParams)

        // Cancel to clean up
        await demo.cancel(stream.id)
    }

    func testUpdateNonExistentGeneration() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        let nonExistentID = GenerationID()
        let newParams = GenerationParams(temperature: 0.9)

        // Should not crash when updating non-existent generation
        demo.update(id: nonExistentID, newParams)

        // Verify no generation info exists
        let info = demo.getGenerationInfo(nonExistentID)
        XCTAssertNil(info)
    }

    // MARK: - Cancellation Tests

    func testCancelGeneration() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        let stream = demo.start(prompt: "Test", params: GenerationParams(maxTokens: 10))

        // Cancel the generation
        await demo.cancel(stream.id)

        // Verify generation info is removed
        let info = demo.getGenerationInfo(stream.id)
        XCTAssertNil(info)
    }

    func testCancelNonExistentGeneration() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        let nonExistentID = GenerationID()

        // Should not crash when cancelling non-existent generation
        await demo.cancel(nonExistentID)

        // Verify no generation info exists
        let info = demo.getGenerationInfo(nonExistentID)
        XCTAssertNil(info)
    }

    func testCancelAllGenerations() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        // Start multiple generations
        let stream1 = demo.start(prompt: "First", params: GenerationParams(maxTokens: 10))
        let stream2 = demo.start(prompt: "Second", params: GenerationParams(maxTokens: 10))
        let stream3 = demo.start(prompt: "Third", params: GenerationParams(maxTokens: 10))

        // Cancel all
        await demo.cancelAll()

        // Verify all generation info is removed
        XCTAssertNil(demo.getGenerationInfo(stream1.id))
        XCTAssertNil(demo.getGenerationInfo(stream2.id))
        XCTAssertNil(demo.getGenerationInfo(stream3.id))

        // Verify no active generations
        XCTAssertEqual(demo.getActiveGenerationIDs().count, 0)
    }

    // MARK: - Generation Info Tests

    func testGetGenerationInfo() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        let params = GenerationParams(seed: 123, topK: 30, temperature: 0.6, maxTokens: 3)
        let stream = demo.start(prompt: "Test", params: params)

        // Get generation info
        let info = demo.getGenerationInfo(stream.id)

        // Verify info is correct
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.params, params)
        XCTAssertNotNil(info?.startTime)

        // Verify start time is recent
        let timeDifference = Date().timeIntervalSince(info?.startTime ?? Date())
        XCTAssertLessThan(timeDifference, 1.0, "Start time should be recent")

        // Cancel to clean up
        await demo.cancel(stream.id)
    }

    func testGetActiveGenerationIDs() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        // Start multiple generations
        let stream1 = demo.start(prompt: "First", params: GenerationParams(maxTokens: 10))
        let stream2 = demo.start(prompt: "Second", params: GenerationParams(maxTokens: 10))
        let stream3 = demo.start(prompt: "Third", params: GenerationParams(maxTokens: 10))

        // Get active IDs
        let activeIDs = demo.getActiveGenerationIDs()

        // Verify all IDs are present
        XCTAssertEqual(activeIDs.count, 3)
        XCTAssertTrue(activeIDs.contains(stream1.id))
        XCTAssertTrue(activeIDs.contains(stream2.id))
        XCTAssertTrue(activeIDs.contains(stream3.id))

        // Cancel one generation
        await demo.cancel(stream2.id)

        // Verify only 2 remain active
        let remainingIDs = demo.getActiveGenerationIDs()
        XCTAssertEqual(remainingIDs.count, 2)
        XCTAssertTrue(remainingIDs.contains(stream1.id))
        XCTAssertTrue(remainingIDs.contains(stream3.id))
        XCTAssertFalse(remainingIDs.contains(stream2.id))

        // Cancel remaining
        await demo.cancelAll()
    }

    // MARK: - Async Stream Context Test (The Key Test)

    func testAsyncStreamContextSolution() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        // This test demonstrates that the async stream context issue is solved
        // The key is that we can iterate over the stream without actor context violations

        let prompt = "Hello world"
        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start generation
        let stream = demo.start(prompt: prompt, params: params)

        // This should work without actor context issues
        // The stream was created within the actor context, so we can iterate over it
        var tokens: [String] = []

        // This is the key test - iterating over the stream should work
        for try await token in stream.stream {
            tokens.append(token)
        }

        // Verify we got some tokens
        XCTAssertFalse(tokens.isEmpty, "Should generate mock tokens")
        XCTAssertGreaterThan(tokens.count, 0)

        print("✅ Async stream context test passed!")
        print("   - Stream created successfully")
        print("   - Stream iteration completed without actor context violations")
        print("   - Tokens collected: \(tokens.count)")
        print("   - Tokens: \(tokens)")

        // The important thing is that this didn't crash with actor isolation errors
        XCTAssertTrue(true, "Async stream context solution works")
    }

    // MARK: - Concurrency Test

    func testConcurrentGenerations() async throws {
        guard let demo else {
            XCTFail("SwiftyCoreLlamaDemo not initialized")
            return
        }

        // Start multiple concurrent generations
        async let generation1 = startAndCollect(demo: demo, prompt: "First generation")
        async let generation2 = startAndCollect(demo: demo, prompt: "Second generation")
        async let generation3 = startAndCollect(demo: demo, prompt: "Third generation")

        // Wait for all to complete
        let (tokens1, tokens2, tokens3) = try await (generation1, generation2, generation3)

        // Verify all completed successfully
        XCTAssertFalse(tokens1.isEmpty)
        XCTAssertFalse(tokens2.isEmpty)
        XCTAssertFalse(tokens3.isEmpty)

        print("✅ Concurrent generations test passed!")
        print("   - Generation 1 tokens: \(tokens1.count)")
        print("   - Generation 2 tokens: \(tokens2.count)")
        print("   - Generation 3 tokens: \(tokens3.count)")

        // The important thing is that concurrent access works without actor isolation issues
        XCTAssertTrue(true, "Concurrent generations work without actor context violations")
    }

    // MARK: - Helper Methods

    private func startAndCollect(demo: SwiftyCoreLlamaDemo, prompt: String) async throws -> [String] {
        let stream = demo.start(prompt: prompt, params: GenerationParams(maxTokens: 5))

        var tokens: [String] = []
        for try await token in stream.stream {
            tokens.append(token)
        }
        return tokens
    }
}
