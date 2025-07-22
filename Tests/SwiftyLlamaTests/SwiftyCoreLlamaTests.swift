import Foundation
import SLlama
import TestUtilities
import XCTest
@testable import SwiftyLlama

@SLlamaActor
final class SwiftyCoreLlamaTests: XCTestCase, @unchecked Sendable {
    // MARK: - Test Properties

    // MARK: - Setup and Teardown

    override func setUp() async throws {
        try await super.setUp()

        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }
    }

    override func tearDown() async throws {
        try await super.tearDown()
    }

    // MARK: - Compilation Tests

    func testCompilation() async throws {
        // Just test that we can reference the type
        let _: SwiftyCoreLlama.Type = SwiftyCoreLlama.self

        XCTAssertTrue(true)
    }

    // MARK: - Initialization Tests

    func testInitialization() async throws {
        // Create SwiftyCoreLlama with real model
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        XCTAssertNotNil(swiftyCore)

        // Test model info
        let modelInfo = swiftyCore.modelInfo
        XCTAssertFalse(modelInfo.name.isEmpty)
        XCTAssertGreaterThan(modelInfo.contextSize, 0)

        // Test vocab info
        let vocabInfo = swiftyCore.vocabInfo
        XCTAssertGreaterThan(vocabInfo.size, 0)
    }

    // MARK: - Basic Functionality Tests

    func testStartGeneration() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let prompt = "Hello"
        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start generation
        let stream = await swiftyCore.start(prompt: prompt, params: params)

        // Verify stream was created
        XCTAssertNotNil(stream)
        XCTAssertNotNil(stream.id)
        XCTAssertNotNil(stream.stream)

        // Verify generation info exists
        let info = await swiftyCore.getGenerationInfo(stream.id)
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.params, params)

        // Cancel to clean up
        await swiftyCore.cancel(stream.id)
    }

    // MARK: - Multiple Generations Tests

    func testMultipleGenerations() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let prompts = ["Hello", "How are", "The weather"]
        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start multiple generations concurrently
        let stream1 = await swiftyCore.start(prompt: prompts[0], params: params)
        let stream2 = await swiftyCore.start(prompt: prompts[1], params: params)
        let stream3 = await swiftyCore.start(prompt: prompts[2], params: params)

        // Verify all streams were created
        XCTAssertNotNil(stream1)
        XCTAssertNotNil(stream2)
        XCTAssertNotNil(stream3)

        // Verify all generation IDs are different
        XCTAssertNotEqual(stream1.id, stream2.id)
        XCTAssertNotEqual(stream2.id, stream3.id)
        XCTAssertNotEqual(stream1.id, stream3.id)

        // Verify all generation info exists
        let info1 = await swiftyCore.getGenerationInfo(stream1.id)
        let info2 = await swiftyCore.getGenerationInfo(stream2.id)
        let info3 = await swiftyCore.getGenerationInfo(stream3.id)

        XCTAssertNotNil(info1)
        XCTAssertNotNil(info2)
        XCTAssertNotNil(info3)

        // Verify active generation IDs
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertEqual(activeIDs.count, 3)
        XCTAssertTrue(activeIDs.contains(stream1.id))
        XCTAssertTrue(activeIDs.contains(stream2.id))
        XCTAssertTrue(activeIDs.contains(stream3.id))

        // Cancel all to clean up
        await swiftyCore.cancelAll()

        // Verify all are cancelled
        let finalActiveIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertEqual(finalActiveIDs.count, 0)
    }

    // MARK: - Conversation Management Tests

    func testConversationManagement() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        // Test starting new conversation
        let conversationId1 = swiftyCore.startNewConversation()
        XCTAssertNotNil(conversationId1)

        // Test getting current conversation
        let currentId = swiftyCore.getCurrentConversationId()
        XCTAssertEqual(currentId, conversationId1)

        // Test conversation info
        let info1 = swiftyCore.getConversationInfo(conversationId1)
        XCTAssertNotNil(info1)
        XCTAssertEqual(info1?.messageCount, 0)
        XCTAssertEqual(info1?.totalTokens, 0)

        // Test starting another conversation
        let conversationId2 = swiftyCore.startNewConversation()
        XCTAssertNotEqual(conversationId1, conversationId2)

        // Test continuing conversation
        try swiftyCore.continueConversation(conversationId1)
        let currentIdAfterContinue = swiftyCore.getCurrentConversationId()
        XCTAssertEqual(currentIdAfterContinue, conversationId1)

        // Test clearing conversation
        swiftyCore.clearConversation(conversationId1)
        let infoAfterClear = swiftyCore.getConversationInfo(conversationId1)
        XCTAssertNil(infoAfterClear)
    }

    func testConversationContinuity() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start a conversation
        let conversationId = swiftyCore.startNewConversation()

        // First message
        let stream1 = await swiftyCore.start(
            prompt: "Hello, my name is Alice",
            params: params,
            conversationId: conversationId
        )

        // Wait for generation to complete
        var response1 = ""
        for try await token in stream1.stream {
            response1 += token
        }

        // Verify conversation was stored
        let info1 = swiftyCore.getConversationInfo(conversationId)
        XCTAssertNotNil(info1)
        XCTAssertGreaterThan(info1?.messageCount ?? 0, 0)
        XCTAssertGreaterThan(info1?.totalTokens ?? 0, 0)

        // Continue the same conversation
        let stream2 = await swiftyCore.start(
            prompt: "What's my name?",
            params: params,
            conversationId: conversationId
        )

        // Wait for generation to complete
        var response2 = ""
        for try await token in stream2.stream {
            response2 += token
        }

        // Verify conversation has more messages
        let info2 = swiftyCore.getConversationInfo(conversationId)
        XCTAssertNotNil(info2)
        XCTAssertGreaterThan(info2?.messageCount ?? 0, info1?.messageCount ?? 0)
        XCTAssertGreaterThan(info2?.totalTokens ?? 0, info1?.totalTokens ?? 0)

        // Clean up
        await swiftyCore.cancelAll()
    }

    func testNewConversationVsContinuation() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // First conversation
        let conversationId1 = swiftyCore.startNewConversation()
        let stream1 = await swiftyCore.start(
            prompt: "Remember: I like pizza",
            params: params,
            conversationId: conversationId1
        )

        // Wait for completion
        for try await _ in stream1.stream {}

        // Second conversation (should be separate)
        let conversationId2 = swiftyCore.startNewConversation()
        let stream2 = await swiftyCore.start(
            prompt: "What do I like?",
            params: params,
            conversationId: conversationId2
        )

        // Wait for completion
        for try await _ in stream2.stream {}

        // Verify they are separate conversations
        XCTAssertNotEqual(conversationId1, conversationId2)

        let info1 = swiftyCore.getConversationInfo(conversationId1)
        let info2 = swiftyCore.getConversationInfo(conversationId2)

        XCTAssertNotNil(info1)
        XCTAssertNotNil(info2)
        // Both conversations should have 1 message each, but they should be separate
        XCTAssertEqual(info1?.messageCount, 1)
        XCTAssertEqual(info2?.messageCount, 1)
        // Verify they are different conversation IDs
        XCTAssertNotEqual(info1?.id, info2?.id)

        // Clean up
        await swiftyCore.cancelAll()
    }
}

// MARK: - Cancellation Tests

extension SwiftyCoreLlamaTests {
    func testIndividualCancellation() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 10)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Hello world", params: params)

        // Verify it's active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertTrue(activeIDs.contains(stream.id))

        // Cancel the specific generation
        await swiftyCore.cancel(stream.id)

        // Verify it's no longer active
        let finalActiveIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertFalse(finalActiveIDs.contains(stream.id))

        // Verify generation info is nil after cancellation (since it's removed from activeGenerations)
        let info = await swiftyCore.getGenerationInfo(stream.id)
        XCTAssertNil(info)
    }

    func testCancellationOfNonExistentGeneration() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        // Create a fake generation ID
        let fakeID = GenerationID()

        // This should not crash and should be a no-op
        await swiftyCore.cancel(fakeID)

        // Verify no generations are active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertEqual(activeIDs.count, 0)
    }

    func testCancellationDuringStreamConsumption() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 20)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Generate a long response", params: params)

        var tokenCount = 0
        var wasCancelled = false

        // Start consuming the stream
        do {
            for try await token in stream.stream {
                tokenCount += 1

                // Cancel after receiving a few tokens
                if tokenCount == 3 {
                    await swiftyCore.cancel(stream.id)
                    wasCancelled = true
                }

                // The stream should eventually terminate after cancellation
                if tokenCount > 10 {
                    break
                }
            }
        } catch {
            // Stream might throw an error after cancellation
            wasCancelled = true
        }

        // Verify cancellation happened
        XCTAssertTrue(wasCancelled)

        // Verify generation is no longer active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertFalse(activeIDs.contains(stream.id))
    }

    func testCancellationOfAlreadyCancelledGeneration() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Hello", params: params)

        // Cancel it once
        await swiftyCore.cancel(stream.id)

        // Cancel it again (should be a no-op)
        await swiftyCore.cancel(stream.id)

        // Verify it's still not active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertFalse(activeIDs.contains(stream.id))
    }

    func testCancelAllWithMultipleGenerations() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start multiple generations
        let stream1 = await swiftyCore.start(prompt: "First", params: params)
        let stream2 = await swiftyCore.start(prompt: "Second", params: params)
        let stream3 = await swiftyCore.start(prompt: "Third", params: params)

        // Verify all are active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertEqual(activeIDs.count, 3)
        XCTAssertTrue(activeIDs.contains(stream1.id))
        XCTAssertTrue(activeIDs.contains(stream2.id))
        XCTAssertTrue(activeIDs.contains(stream3.id))

        // Cancel all
        await swiftyCore.cancelAll()

        // Verify none are active
        let finalActiveIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertEqual(finalActiveIDs.count, 0)

        // Verify all generation info are nil after cancellation (since they're removed from activeGenerations)
        let info1 = await swiftyCore.getGenerationInfo(stream1.id)
        let info2 = await swiftyCore.getGenerationInfo(stream2.id)
        let info3 = await swiftyCore.getGenerationInfo(stream3.id)

        XCTAssertNil(info1)
        XCTAssertNil(info2)
        XCTAssertNil(info3)
    }

    func testCancellationWithOnTerminationCallback() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 10)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Test termination", params: params)

        // Verify it's active
        let info = await swiftyCore.getGenerationInfo(stream.id)
        XCTAssertNotNil(info)
        XCTAssertTrue(info?.isActive ?? false)

        // Cancel the generation
        await swiftyCore.cancel(stream.id)

        // The onTermination callback should trigger another cancel call
        // This should be handled gracefully (no-op for already cancelled)

        // Verify it's no longer active (should be nil since removed from activeGenerations)
        let finalInfo = await swiftyCore.getGenerationInfo(stream.id)
        XCTAssertNil(finalInfo)
    }

    func testCancellationStateVerification() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Test state", params: params)

        // Verify initial state
        let initialInfo = await swiftyCore.getGenerationInfo(stream.id)
        XCTAssertNotNil(initialInfo)
        XCTAssertTrue(initialInfo?.isActive ?? false)
        XCTAssertEqual(initialInfo?.id, stream.id)

        // Cancel it
        await swiftyCore.cancel(stream.id)

        // Verify cancelled state (should be nil since removed from activeGenerations)
        let cancelledInfo = await swiftyCore.getGenerationInfo(stream.id)
        XCTAssertNil(cancelledInfo)

        // Verify it's not in active list
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertFalse(activeIDs.contains(stream.id))
    }

    func testCancellationWithEmptyActiveGenerations() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        // Cancel all when no generations are active
        await swiftyCore.cancelAll()

        // Verify no active generations
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertEqual(activeIDs.count, 0)
    }
}
