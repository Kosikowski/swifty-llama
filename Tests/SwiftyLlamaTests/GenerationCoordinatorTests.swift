import Foundation
import XCTest
@testable import SwiftyLlama

@SwiftyLlamaCore
final class GenerationCoordinatorTests: XCTestCase {
    // MARK: - Test Properties

    private var coordinator: GenerationCoordinator?
    private var mockCore: MockLlamaCoreActor?

    // MARK: - Setup and Teardown

    override func setUp() async throws {
        try await super.setUp()

        // Create mock core and coordinator
        mockCore = MockLlamaCoreActor()
        coordinator = GenerationCoordinator(core: mockCore!)
    }

    override func tearDown() async throws {
        coordinator = nil
        mockCore = nil
        try await super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInitialization() {
        XCTAssertNotNil(coordinator)
        XCTAssertNotNil(mockCore)
    }

    // MARK: - Generation Start Tests

    func testStartGeneration() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        let prompt = "Hello world"
        let params = GenerationParams(temperature: 0.7)

        // Start generation
        let stream = await coordinator.start(prompt: prompt, params: params)

        // Verify stream was created
        XCTAssertNotNil(stream)
        XCTAssertNotNil(stream.id)
        XCTAssertNotNil(stream.stream)

        // Verify generation was started in core
        XCTAssertEqual(mockCore?.startedGenerations.count, 1)
        XCTAssertEqual(mockCore?.startedGenerations.first?.prompt, prompt)
        XCTAssertEqual(mockCore?.startedGenerations.first?.params, params)
    }

    func testStartMultipleGenerations() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        let prompts = ["First", "Second", "Third"]
        let params = GenerationParams(temperature: 0.5)

        var streams: [GenerationStream] = []

        // Start multiple generations
        for prompt in prompts {
            let stream = await coordinator.start(prompt: prompt, params: params)
            streams.append(stream)
        }

        // Verify all generations were started
        XCTAssertEqual(streams.count, 3)
        XCTAssertEqual(mockCore?.startedGenerations.count, 3)

        // Verify all have unique IDs
        let ids = streams.map(\.id)
        XCTAssertEqual(Set(ids).count, 3, "All generation IDs should be unique")
    }

    // MARK: - Parameter Update Tests

    func testUpdateGenerationParameters() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        let initialParams = GenerationParams(temperature: 0.5, topK: 20)
        let updatedParams = GenerationParams(temperature: 0.8, topK: 40)

        // Start generation
        let stream = coordinator.start(prompt: "Test", params: initialParams)

        // Update parameters
        coordinator.update(id: stream.id, updatedParams)

        // Verify parameters were updated
        let info = coordinator.getGenerationInfo(stream.id)
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.params, updatedParams)
    }

    func testUpdateNonExistentGeneration() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        let nonExistentID = GenerationID()
        let newParams = GenerationParams(temperature: 0.9)

        // Should not crash when updating non-existent generation
        await coordinator.update(id: nonExistentID, newParams)

        // Verify no generation info exists
        let info = await coordinator.getGenerationInfo(nonExistentID)
        XCTAssertNil(info)
    }

    // MARK: - Cancellation Tests

    func testCancelGeneration() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        let stream = await coordinator.start(prompt: "Test", params: GenerationParams())

        // Cancel the generation
        await coordinator.cancel(stream.id)

        // Verify generation was cancelled in core
        XCTAssertEqual(mockCore?.cancelledGenerations.count, 1)
        XCTAssertEqual(mockCore?.cancelledGenerations.first, stream.id)

        // Verify generation info is removed
        let info = await coordinator.getGenerationInfo(stream.id)
        XCTAssertNil(info)
    }

    func testCancelNonExistentGeneration() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        let nonExistentID = GenerationID()

        // Should not crash when cancelling non-existent generation
        await coordinator.cancel(nonExistentID)

        // Verify no cancellation was recorded
        XCTAssertEqual(mockCore?.cancelledGenerations.count, 0)
    }

    func testCancelAllGenerations() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        // Start multiple generations
        let stream1 = await coordinator.start(prompt: "First", params: GenerationParams())
        let stream2 = await coordinator.start(prompt: "Second", params: GenerationParams())
        let stream3 = await coordinator.start(prompt: "Third", params: GenerationParams())

        // Cancel all
        await coordinator.cancelAll()

        // Verify all were cancelled
        XCTAssertEqual(mockCore?.cancelledGenerations.count, 3)
        XCTAssertTrue(mockCore?.cancelledGenerations.contains(stream1.id) == true)
        XCTAssertTrue(mockCore?.cancelledGenerations.contains(stream2.id) == true)
        XCTAssertTrue(mockCore?.cancelledGenerations.contains(stream3.id) == true)

        // Verify all generation info is removed
        XCTAssertNil(coordinator.getGenerationInfo(stream1.id))
        XCTAssertNil(coordinator.getGenerationInfo(stream2.id))
        XCTAssertNil(coordinator.getGenerationInfo(stream3.id))
    }

    // MARK: - Generation Info Tests

    func testGetGenerationInfo() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        let params = GenerationParams(temperature: 0.6, topK: 30)
        let stream = coordinator.start(prompt: "Test", params: params)

        // Get generation info
        let info = coordinator.getGenerationInfo(stream.id)

        // Verify info is correct
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.params, params)
        XCTAssertNotNil(info?.startTime)

        // Verify start time is recent
        let timeDifference = Date().timeIntervalSince(info?.startTime ?? Date())
        XCTAssertLessThan(timeDifference, 1.0, "Start time should be recent")
    }

    func testGetActiveGenerationIDs() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        // Start multiple generations
        let stream1 = coordinator.start(prompt: "First", params: GenerationParams())
        let stream2 = coordinator.start(prompt: "Second", params: GenerationParams())
        let stream3 = coordinator.start(prompt: "Third", params: GenerationParams())

        // Get active IDs
        let activeIDs = await coordinator.getActiveGenerationIDs()

        // Verify all IDs are present
        XCTAssertEqual(activeIDs.count, 3)
        XCTAssertTrue(activeIDs.contains(stream1.id))
        XCTAssertTrue(activeIDs.contains(stream2.id))
        XCTAssertTrue(activeIDs.contains(stream3.id))

        // Cancel one generation
        await coordinator.cancel(stream2.id)

        // Verify only 2 remain active
        let remainingIDs = await coordinator.getActiveGenerationIDs()
        XCTAssertEqual(remainingIDs.count, 2)
        XCTAssertTrue(remainingIDs.contains(stream1.id))
        XCTAssertTrue(remainingIDs.contains(stream3.id))
        XCTAssertFalse(remainingIDs.contains(stream2.id))
    }

    // MARK: - Stream Handling Tests

    func testStreamCompletion() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        // Configure mock to complete successfully
        mockCore?.shouldCompleteSuccessfully = true
        mockCore?.mockTokens = ["Hello", " world", "!"]

        let stream = await coordinator.start(prompt: "Test", params: GenerationParams())

        // Collect tokens
        var tokens: [String] = []
        for try await token in stream.stream {
            tokens.append(token)
        }

        // Verify tokens were received
        XCTAssertEqual(tokens, ["Hello", " world", "!"])

        // Verify generation info is cleaned up
        let info = await coordinator.getGenerationInfo(stream.id)
        XCTAssertNil(info)
    }

    func testStreamError() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        // Configure mock to throw error
        mockCore?.shouldThrowError = true
        mockCore?.mockError = GenerationError.internalFailure("Test error")

        let stream = await coordinator.start(prompt: "Test", params: GenerationParams())

        // Verify error is propagated
        do {
            for try await _ in stream.stream {
                // Should not reach here
                XCTFail("Should not receive tokens when error occurs")
            }
        } catch {
            XCTAssertTrue(error is GenerationError)
            XCTAssertEqual((error as? GenerationError)?.localizedDescription, "Internal failure: Test error")
        }

        // Verify generation info is cleaned up
        let info = await coordinator.getGenerationInfo(stream.id)
        XCTAssertNil(info)
    }

    func testStreamCancellation() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        // Configure mock to be cancellable
        mockCore?.shouldBeCancellable = true

        let stream = await coordinator.start(prompt: "Test", params: GenerationParams())

        // Cancel immediately
        await coordinator.cancel(stream.id)

        // Verify cancellation error is propagated
        do {
            for try await _ in stream.stream {
                XCTFail("Should not receive tokens when cancelled")
            }
        } catch {
            XCTAssertTrue(error is GenerationError)
            XCTAssertEqual((error as? GenerationError)?.localizedDescription, "Generation was aborted by user")
        }
    }

    // MARK: - Parameter Validation Tests

    func testGenerationWithExtremeParameters() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        let extremeParams = [
            GenerationParams(temperature: 0.0, topK: 1, topP: 0.1),
            GenerationParams(temperature: 2.0, topK: 100, topP: 1.0),
            GenerationParams(repeatPenalty: 2.0, repetitionLookback: 128),
        ]

        for params in extremeParams {
            let stream = coordinator.start(prompt: "Test", params: params)

            // Verify generation was started with extreme parameters
            XCTAssertNotNil(stream)

            // Cancel to clean up
            await coordinator.cancel(stream.id)
        }
    }

    // MARK: - Concurrency Tests

    func testConcurrentGenerations() async throws {
        guard let coordinator else {
            XCTFail("Coordinator not initialized")
            return
        }

        // Configure mock for concurrent operations
        mockCore?.shouldCompleteSuccessfully = true
        mockCore?.mockTokens = ["Token"]

        // Start multiple concurrent generations
        async let generation1 = startAndCollect(coordinator: coordinator, prompt: "First")
        async let generation2 = startAndCollect(coordinator: coordinator, prompt: "Second")
        async let generation3 = startAndCollect(coordinator: coordinator, prompt: "Third")

        // Wait for all to complete
        let (tokens1, tokens2, tokens3) = try await (generation1, generation2, generation3)

        // Verify all completed successfully
        XCTAssertFalse(tokens1.isEmpty)
        XCTAssertFalse(tokens2.isEmpty)
        XCTAssertFalse(tokens3.isEmpty)
    }

    // MARK: - Helper Methods

    private func startAndCollect(coordinator: GenerationCoordinator, prompt: String) async throws -> [String] {
        let stream = await coordinator.start(prompt: prompt, params: GenerationParams())

        var tokens: [String] = []
        for try await token in stream.stream {
            tokens.append(token)
        }

        return tokens
    }
}

// MARK: - Mock LlamaCoreActor

private class MockLlamaCoreActor: SwiftyLlamaCore {
    var startedGenerations: [(id: GenerationID, prompt: String, params: GenerationParams)] = []
    var cancelledGenerations: [GenerationID] = []
    var shouldCompleteSuccessfully = false
    var shouldThrowError = false
    var shouldBeCancellable = false
    var mockTokens: [String] = []
    var mockError: Error?

    override func generate(
        id: GenerationID,
        prompt: String,
        params: GenerationParams
    ) async throws
        -> AsyncThrowingStream<String, Error>
    {
        startedGenerations.append((id: id, prompt: prompt, params: params))

        return AsyncThrowingStream { continuation in
            Task {
                if self.shouldThrowError {
                    continuation.finish(throwing: self.mockError ?? GenerationError.internalFailure("Mock error"))
                    return
                }

                if self.shouldBeCancellable {
                    // Simulate cancellation
                    continuation.finish(throwing: GenerationError.abortedByUser)
                    return
                }

                if self.shouldCompleteSuccessfully {
                    for token in self.mockTokens {
                        continuation.yield(token)
                    }
                    continuation.finish()
                } else {
                    // Default behavior - yield one token then finish
                    continuation.yield("Mock token")
                    continuation.finish()
                }
            }
        }
    }

    override func cancel(id: GenerationID) {
        cancelledGenerations.append(id)
    }
}
