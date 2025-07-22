import Foundation
import SLlama
import Testing
import TestUtilities
@testable import SwiftyLlama

@SwiftyLlamaActor
struct SwiftyCoreLlamaTests {
    // MARK: - Compilation Tests

    @Test("SwiftyCoreLlama compilation test")
    func compilation() throws {
        // Just test that we can reference the type
        let _: SwiftyCoreLlama.Type = SwiftyCoreLlama.self
        #expect(true, "Compilation test should pass")
    }

    // MARK: - Initialization Tests

    @Test("SwiftyCoreLlama initialization test")
    func initialization() async throws {
        // Fail if model not available
        #expect(TestUtilities.isTestModelAvailable(), "Test model must be available for initialization test")

        // Create SwiftyCoreLlama with real model
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        // Test model info
        let modelInfo = swiftyCore.modelInfo
        #expect(!modelInfo.name.isEmpty, "Model name should not be empty")
        #expect(modelInfo.contextSize > 0, "Context size should be positive")

        // Test vocab info
        let vocabInfo = swiftyCore.vocabInfo
        #expect(vocabInfo.size > 0, "Vocab size should be positive")
    }

    // MARK: - Basic Functionality Tests

    @Test("SwiftyCoreLlama start generation test")
    func startGeneration() async throws {
        // Fail if model not available
        #expect(TestUtilities.isTestModelAvailable(), "Test model must be available for generation test")

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let prompt = "Hello"
        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start generation
        let stream = await swiftyCore.start(prompt: prompt, params: params)

        // Verify generation info exists
        let info = await swiftyCore.getGenerationInfo(stream.id)
        #expect(info != nil, "Generation info should exist")
        #expect(info?.params == params, "Generation params should match")

        // Cancel to clean up
        await swiftyCore.cancel(stream.id)
    }

    // MARK: - Multiple Generations Tests

    @Test("SwiftyCoreLlama multiple generations test")
    func multipleGenerations() async throws {
        // Fail if model not available
        #expect(TestUtilities.isTestModelAvailable(), "Test model must be available for multiple generations test")

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let prompts = ["Hello", "How are", "The weather"]
        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start multiple generations concurrently
        let stream1 = await swiftyCore.start(prompt: prompts[0], params: params)
        let stream2 = await swiftyCore.start(prompt: prompts[1], params: params)
        let stream3 = await swiftyCore.start(prompt: prompts[2], params: params)

        // Verify all generation IDs are different
        #expect(stream1.id != stream2.id, "Generation IDs should be different")
        #expect(stream2.id != stream3.id, "Generation IDs should be different")
        #expect(stream1.id != stream3.id, "Generation IDs should be different")

        // Verify all generation info exists
        let info1 = await swiftyCore.getGenerationInfo(stream1.id)
        let info2 = await swiftyCore.getGenerationInfo(stream2.id)
        let info3 = await swiftyCore.getGenerationInfo(stream3.id)

        #expect(info1 != nil, "Generation info 1 should exist")
        #expect(info2 != nil, "Generation info 2 should exist")
        #expect(info3 != nil, "Generation info 3 should exist")

        // Verify active generation IDs
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(activeIDs.count == 3, "Should have 3 active generations")
        #expect(activeIDs.contains(stream1.id), "Should contain stream1 ID")
        #expect(activeIDs.contains(stream2.id), "Should contain stream2 ID")
        #expect(activeIDs.contains(stream3.id), "Should contain stream3 ID")

        // Cancel all to clean up
        await swiftyCore.cancelAll()

        // Verify all are cancelled
        let finalActiveIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(finalActiveIDs.count == 0, "Should have no active generations after cancellation")
    }

    // MARK: - Conversation Management Tests

    @Test("SwiftyCoreLlama conversation management test")
    func conversationManagement() async throws {
        // Fail if model not available
        #expect(TestUtilities.isTestModelAvailable(), "Test model must be available for conversation management test")

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        // Test starting new conversation
        let conversationId1 = swiftyCore.startNewConversation()

        // Test getting current conversation
        let currentId = swiftyCore.getCurrentConversationId()
        #expect(currentId == conversationId1, "Current conversation ID should match")

        // Test conversation info
        let info1 = swiftyCore.getConversationInfo(conversationId1)
        #expect(info1 != nil, "Conversation info should exist")
        #expect(info1?.messageCount == 0, "New conversation should have 0 messages")
        #expect(info1?.totalTokens == 0, "New conversation should have 0 tokens")

        // Test starting another conversation
        let conversationId2 = swiftyCore.startNewConversation()
        #expect(conversationId1 != conversationId2, "Conversation IDs should be different")

        // Test continuing conversation
        try swiftyCore.continueConversation(conversationId1)
        let currentIdAfterContinue = swiftyCore.getCurrentConversationId()
        #expect(currentIdAfterContinue == conversationId1, "Current conversation should be set correctly")

        // Test clearing conversation
        swiftyCore.clearConversation(conversationId1)
        let infoAfterClear = swiftyCore.getConversationInfo(conversationId1)
        #expect(infoAfterClear == nil, "Cleared conversation should return nil")
    }

    @Test("SwiftyCoreLlama conversation continuity test")
    func conversationContinuity() async throws {
        // Fail if model not available
        #expect(TestUtilities.isTestModelAvailable(), "Test model must be available for conversation continuity test")

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
        #expect(info1 != nil, "Conversation info should exist")
        #expect(info1?.messageCount ?? 0 > 0, "Conversation should have messages")
        #expect(info1?.totalTokens ?? 0 > 0, "Conversation should have tokens")

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
        #expect(info2 != nil, "Conversation info should still exist")
        #expect(info2?.messageCount ?? 0 > info1?.messageCount ?? 0, "Conversation should have more messages")
        #expect(info2?.totalTokens ?? 0 > info1?.totalTokens ?? 0, "Conversation should have more tokens")

        // Clean up
        await swiftyCore.cancelAll()
    }

    @Test("SwiftyCoreLlama new conversation vs continuation test")
    func newConversationVsContinuation() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for new conversation vs continuation test"
        )

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
        #expect(conversationId1 != conversationId2, "Conversation IDs should be different")

        let info1 = swiftyCore.getConversationInfo(conversationId1)
        let info2 = swiftyCore.getConversationInfo(conversationId2)

        #expect(info1 != nil, "First conversation info should exist")
        #expect(info2 != nil, "Second conversation info should exist")
        // Both conversations should have 2 messages each (user message + assistant response), but they should be
        // separate
        #expect(info1?.messageCount == 2, "First conversation should have 2 messages (user + assistant)")
        #expect(info2?.messageCount == 2, "Second conversation should have 2 messages (user + assistant)")
        // Verify they are different conversation IDs
        #expect(info1?.id != info2?.id, "Conversation IDs should be different")

        // Clean up
        await swiftyCore.cancelAll()
    }
}

// MARK: - Cancellation Tests

extension SwiftyCoreLlamaTests {
    @Test("SwiftyCoreLlama individual cancellation test")
    func individualCancellation() async throws {
        // Fail if model not available
        #expect(TestUtilities.isTestModelAvailable(), "Test model must be available for individual cancellation test")

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 10)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Hello world", params: params)

        // Verify it's active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(activeIDs.contains(stream.id), "Generation should be active")

        // Cancel the specific generation
        await swiftyCore.cancel(stream.id)

        // Verify it's no longer active
        let finalActiveIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(!finalActiveIDs.contains(stream.id), "Generation should not be active after cancellation")

        // Verify generation info is nil after cancellation (since it's removed from activeGenerations)
        let info = await swiftyCore.getGenerationInfo(stream.id)
        #expect(info == nil, "Generation info should be nil after cancellation")
    }

    @Test("SwiftyCoreLlama cancellation of non-existent generation test")
    func cancellationOfNonExistentGeneration() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for cancellation of non-existent generation test"
        )

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        // Create a fake generation ID
        let fakeID = GenerationID()

        // This should not crash and should be a no-op
        await swiftyCore.cancel(fakeID)

        // Verify no generations are active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(activeIDs.count == 0, "Should have no active generations")
    }

    @Test("SwiftyCoreLlama cancellation during stream consumption test")
    func cancellationDuringStreamConsumption() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for cancellation during stream consumption test"
        )

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 20)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Generate a long response", params: params)

        var tokenCount = 0
        var wasCancelled = false
        var tokens: [String] = []

        // Start consuming the stream
        do {
            for try await token in stream.stream {
                tokens.append(token)
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
        #expect(wasCancelled, "Cancellation should have occurred")

        // Verify we received some tokens
        #expect(tokens.count > 0, "Should have received some tokens")

        // Verify generation is no longer active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(!activeIDs.contains(stream.id), "Generation should not be active after cancellation")

        // Print the tokens received during the test
        print("Tokens received during cancellation test: \(tokens)")
    }

    @Test("SwiftyCoreLlama cancellation of already cancelled generation test")
    func cancellationOfAlreadyCancelledGeneration() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for cancellation of already cancelled generation test"
        )

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
        #expect(!activeIDs.contains(stream.id), "Generation should not be active after double cancellation")
    }

    @Test("SwiftyCoreLlama cancel all with multiple generations test")
    func cancelAllWithMultipleGenerations() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for cancel all with multiple generations test"
        )

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start multiple generations
        let stream1 = await swiftyCore.start(prompt: "First", params: params)
        let stream2 = await swiftyCore.start(prompt: "Second", params: params)
        let stream3 = await swiftyCore.start(prompt: "Third", params: params)

        // Verify all are active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(activeIDs.count == 3, "Should have 3 active generations")
        #expect(activeIDs.contains(stream1.id), "Should contain stream1 ID")
        #expect(activeIDs.contains(stream2.id), "Should contain stream2 ID")
        #expect(activeIDs.contains(stream3.id), "Should contain stream3 ID")

        // Cancel all
        await swiftyCore.cancelAll()

        // Verify none are active
        let finalActiveIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(finalActiveIDs.count == 0, "Should have no active generations after cancelAll")

        // Verify all generation info are nil after cancellation (since they're removed from activeGenerations)
        let info1 = await swiftyCore.getGenerationInfo(stream1.id)
        let info2 = await swiftyCore.getGenerationInfo(stream2.id)
        let info3 = await swiftyCore.getGenerationInfo(stream3.id)

        #expect(info1 == nil, "Generation info 1 should be nil after cancellation")
        #expect(info2 == nil, "Generation info 2 should be nil after cancellation")
        #expect(info3 == nil, "Generation info 3 should be nil after cancellation")
    }

    @Test("SwiftyCoreLlama cancellation with onTermination callback test")
    func cancellationWithOnTerminationCallback() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for cancellation with onTermination callback test"
        )

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 10)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Test termination", params: params)

        // Verify it's active
        let info = await swiftyCore.getGenerationInfo(stream.id)
        #expect(info != nil, "Generation info should exist")
        #expect(info?.isActive == true, "Generation should be active")

        // Cancel the generation
        await swiftyCore.cancel(stream.id)

        // The onTermination callback should trigger another cancel call
        // This should be handled gracefully (no-op for already cancelled)

        // Verify it's no longer active (should be nil since removed from activeGenerations)
        let finalInfo = await swiftyCore.getGenerationInfo(stream.id)
        #expect(finalInfo == nil, "Generation info should be nil after cancellation")
    }

    @Test("SwiftyCoreLlama cancellation state verification test")
    func cancellationStateVerification() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for cancellation state verification test"
        )

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Test state", params: params)

        // Verify initial state
        let initialInfo = await swiftyCore.getGenerationInfo(stream.id)
        #expect(initialInfo != nil, "Initial generation info should exist")
        #expect(initialInfo?.isActive == true, "Generation should be active initially")
        #expect(initialInfo?.id == stream.id, "Generation ID should match")

        // Cancel it
        await swiftyCore.cancel(stream.id)

        // Verify cancelled state (should be nil since removed from activeGenerations)
        let cancelledInfo = await swiftyCore.getGenerationInfo(stream.id)
        #expect(cancelledInfo == nil, "Generation info should be nil after cancellation")

        // Verify it's not in active list
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(!activeIDs.contains(stream.id), "Generation should not be in active list after cancellation")
    }

    @Test("SwiftyCoreLlama cancellation with empty active generations test")
    func cancellationWithEmptyActiveGenerations() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for cancellation with empty active generations test"
        )

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        // Cancel all when no generations are active
        await swiftyCore.cancelAll()

        // Verify no active generations
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(activeIDs.count == 0, "Should have no active generations")
    }

    // MARK: - Error Handling Tests

    @Test("SwiftyCoreLlama conversation not found error test")
    func conversationNotFoundError() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let nonExistentConversationId = ConversationID()

        // Try to continue a non-existent conversation
        do {
            try swiftyCore.continueConversation(nonExistentConversationId)
            #expect(Bool(false), "Should have thrown conversationNotFound error")
        } catch let error as GenerationError {
            #expect(error == .conversationNotFound, "Should throw conversationNotFound error")
        } catch {
            #expect(Bool(false), "Should have thrown GenerationError, got: \(error)")
        }
    }

    @Test("SwiftyCoreLlama generation error handling test")
    func generationErrorHandling() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for generation error handling test"
        )

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Test error handling", params: params)

        var receivedError: GenerationError?
        var tokenCount = 0

        // Consume the stream and catch any errors
        do {
            for try await _ in stream.stream {
                tokenCount += 1
                if tokenCount > 10 {
                    break
                }
            }
        } catch let error as GenerationError {
            receivedError = error
        } catch {
            #expect(Bool(false), "Should have thrown GenerationError, got: \(error)")
        }

        // Verify we either got tokens or a specific error
        #expect(tokenCount > 0 || receivedError != nil, "Should have received tokens or a specific error")
    }

    @Test("SwiftyCoreLlama error description test")
    func errorDescriptionTest() {
        // Test that all GenerationError cases have proper descriptions
        let errors: [GenerationError] = [
            .abortedByUser,
            .modelNotLoaded,
            .contextNotInitialized,
            .conversationNotFound,
            .contextPreparationFailed,
            .tokenizationFailed,
            .generationFailed,
            .invalidState,
        ]

        for error in errors {
            let description = error.errorDescription
            #expect(description != nil, "Error description should not be nil for \(error)")
            #expect(!description!.isEmpty, "Error description should not be empty for \(error)")
        }
    }

    // MARK: - Persistence Tests

    @Test("SwiftyCoreLlama conversation persistence test")
    func conversationPersistenceTest() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for conversation persistence test"
        )

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start a conversation and generate some content
        let stream = await swiftyCore.start(prompt: "Hello, how are you?", params: params)

        // Verify conversation was created and generation is active
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(activeIDs.count > 0, "Should have active generations")

        var tokens: [String] = []
        for try await token in stream.stream {
            tokens.append(token)
            if tokens.count >= 3 {
                break
            }
        }

        // Get conversation state
        let conversationState = swiftyCore.getConversationState()
        #expect(conversationState.count > 0, "Should have conversation state")

        // Test JSON serialization/deserialization
        let jsonData = try swiftyCore.saveConversationsToJSON()
        #expect(jsonData.count > 0, "JSON data should not be empty")

        // Create a new instance and restore conversations
        let newSwiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)
        try newSwiftyCore.loadConversationsFromJSON(jsonData)

        // Verify conversations were restored
        let restoredState = newSwiftyCore.getConversationState()
        #expect(restoredState.count == conversationState.count, "Should have same number of conversations")

        // Test continuing a restored conversation
        if let firstConversation = restoredState.first {
            try newSwiftyCore.continueConversationWithWarmUp(firstConversation.id)

            // Start a new generation in the restored conversation
            let newStream = await newSwiftyCore.start(
                prompt: "Continue our conversation",
                params: params,
                conversationId: firstConversation.id
            )

            var newTokens: [String] = []
            for try await token in newStream.stream {
                newTokens.append(token)
                if newTokens.count >= 2 {
                    break
                }
            }

            #expect(newTokens.count > 0, "Should be able to continue restored conversation")
        }
    }

    @Test("SwiftyCoreLlama conversation warm-up test")
    func conversationWarmUpTest() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for conversation warm-up test"
        )

        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start a conversation
        let stream = await swiftyCore.start(prompt: "Tell me a short story", params: params)

        var tokens: [String] = []
        for try await token in stream.stream {
            tokens.append(token)
        }

        // Get conversation state
        let conversationState = swiftyCore.getConversationState()
        #expect(conversationState.count > 0, "Should have conversation state")

        if let conversation = conversationState.first {
            // Test warm-up functionality
            try swiftyCore.continueConversationWithWarmUp(conversation.id)

            // Verify the conversation is now active
            let currentId = swiftyCore.getCurrentConversationId()
            #expect(currentId == conversation.id, "Current conversation should be set")

            // Test that we can continue the warmed-up conversation
            let continueStream = await swiftyCore.start(
                prompt: "Continue the story",
                params: params,
                conversationId: conversation.id
            )

            var continueTokens: [String] = []
            for try await token in continueStream.stream {
                continueTokens.append(token)
                if continueTokens.count >= 2 {
                    break
                }
            }

            #expect(continueTokens.count > 0, "Should be able to continue warmed-up conversation")
        }
    }
}
