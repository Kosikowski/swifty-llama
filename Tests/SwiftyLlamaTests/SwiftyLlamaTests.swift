import Foundation
import SLlama
import Testing
import TestUtilities
@testable import SwiftyLlama

@SwiftyLlamaActor
struct SwiftyLlamaTests {
    // MARK: - Compilation Tests

    @Test("SwiftyCoreLlama compilation test")
    func compilation() throws {
        // Just test that we can reference the type
        let _: SwiftyLlamaCore.Type = SwiftyLlamaCore.self
        #expect(Bool(true), "Compilation test should pass")
    }

    // MARK: - Initialization Tests

    @Test("SwiftyCoreLlama initialization test")
    func initialization() async throws {
        // Fail if model not available
        #expect(TestUtilities.isTestModelAvailable(), "Test model must be available for initialization test")

        // Create SwiftyCoreLlama with real model
        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath, contextSize: 2048)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath, contextSize: 2048)

        let prompt = "Hello"
        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start generation
        let stream = await swiftyCore.start(prompt: prompt, params: params, conversationId: nil)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let prompts = ["Hello", "How are", "The weather"]
        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start multiple generations concurrently
        let stream1 = await swiftyCore.start(prompt: prompts[0], params: params, conversationId: nil)
        let stream2 = await swiftyCore.start(prompt: prompts[1], params: params, conversationId: nil)
        let stream3 = await swiftyCore.start(prompt: prompts[2], params: params, conversationId: nil)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

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

extension SwiftyLlamaTests {
    @Test("SwiftyCoreLlama individual cancellation test")
    func individualCancellation() async throws {
        // Fail if model not available
        #expect(TestUtilities.isTestModelAvailable(), "Test model must be available for individual cancellation test")

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 10)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Hello world", params: params, conversationId: nil)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 20)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Generate a long response", params: params, conversationId: nil)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Hello", params: params, conversationId: nil)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start multiple generations
        let stream1 = await swiftyCore.start(prompt: "First", params: params, conversationId: nil)
        let stream2 = await swiftyCore.start(prompt: "Second", params: params, conversationId: nil)
        let stream3 = await swiftyCore.start(prompt: "Third", params: params, conversationId: nil)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 10)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Test termination", params: params, conversationId: nil)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Test state", params: params, conversationId: nil)

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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        // Cancel all when no generations are active
        await swiftyCore.cancelAll()

        // Verify no active generations
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        #expect(activeIDs.count == 0, "Should have no active generations")
    }

    // MARK: - Error Handling Tests

    @Test("SwiftyCoreLlama conversation not found error test")
    func conversationNotFoundError() async throws {
        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let nonExistentConversationId = ConversationID()

        // Try to continue a non-existent conversation
        do {
            try swiftyCore.continueConversation(nonExistentConversationId)
            #expect(Bool(false), "Should have thrown conversationNotFound error")
        } catch let error as GenerationError {
            // Check if it's a conversationNotFound error (we can't compare directly due to associated values)
            if case .conversationNotFound = error {
                // This is the expected error
            } else {
                #expect(Bool(false), "Should throw conversationNotFound error, got: \(error)")
            }
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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start a generation
        let stream = await swiftyCore.start(prompt: "Test error handling", params: params, conversationId: nil)

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
        let testConversationId = ConversationID()
        let testGenerationId = GenerationID()

        let errors: [GenerationError] = [
            .abortedByUser(generationId: testGenerationId),
            .modelNotLoaded,
            .contextNotInitialized,
            .conversationNotFound(conversationId: testConversationId),
            .contextPreparationFailed(conversationId: testConversationId),
            .tokenizationFailed(conversationId: testConversationId),
            .generationFailed(generationId: testGenerationId),
            .invalidState(conversationId: testConversationId),
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

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start a conversation and generate some content
        let stream = await swiftyCore.start(prompt: "Hello, how are you?", params: params, conversationId: nil)

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
        let conversationState = swiftyCore.getAllConversations()
        #expect(conversationState.count > 0, "Should have conversation state")

        // Test JSON serialization/deserialization
        let jsonData = try await swiftyCore.exportConversations()
        #expect(jsonData.count > 0, "JSON data should not be empty")

        // Create a new instance and restore conversations
        let newSwiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)
        try newSwiftyCore.importConversations(jsonData)

        // Verify conversations were restored
        let restoredState = newSwiftyCore.getAllConversations()
        #expect(restoredState.count == conversationState.count, "Should have same number of conversations")

        // Test continuing a restored conversation
        if let firstConversation = restoredState.first {
            try newSwiftyCore.continueConversationWithContextReconstruction(firstConversation.id)

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

    @Test("SwiftyCoreLlama conversation context reconstruction test")
    func conversationContextReconstructionTest() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for conversation context reconstruction test"
        )

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start a conversation
        let stream = await swiftyCore.start(prompt: "Tell me a short story", params: params, conversationId: nil)

        var tokens: [String] = []
        for try await token in stream.stream {
            tokens.append(token)
        }

        // Get conversation state
        let conversationState = swiftyCore.getAllConversations()
        #expect(conversationState.count > 0, "Should have conversation state")

        if let conversation = conversationState.first {
            // Test context reconstruction functionality
            try swiftyCore.continueConversationWithContextReconstruction(conversation.id)

            // Verify the conversation is now active
            let currentId = swiftyCore.getCurrentConversationId()
            #expect(currentId == conversation.id, "Current conversation should be set")

            // Test that we can continue the conversation with reconstructed context
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

            #expect(continueTokens.count > 0, "Should be able to continue conversation with reconstructed context")
        }
    }

    @Test("SwiftyCoreLlama multi-conversation state isolation test")
    func multiConversationStateIsolationTest() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for multi-conversation state isolation test"
        )

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath)

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Create multiple conversations
        let conversationId1 = swiftyCore.startNewConversation()
        let conversationId2 = swiftyCore.startNewConversation()
        let conversationId3 = swiftyCore.startNewConversation()

        #expect(conversationId1 != conversationId2, "Conversation IDs should be different")
        #expect(conversationId2 != conversationId3, "Conversation IDs should be different")
        #expect(conversationId1 != conversationId3, "Conversation IDs should be different")

        // Start generation in first conversation with short prompt (fewer tokens)
        let stream1 = await swiftyCore.start(
            prompt: "Hi",
            params: params,
            conversationId: conversationId1
        )

        // Wait for completion
        for try await _ in stream1.stream {}

        // Start generation in second conversation with medium prompt (more tokens)
        let stream2 = await swiftyCore.start(
            prompt: "Hello there, how are you doing today? I hope you're having a wonderful day.",
            params: params,
            conversationId: conversationId2
        )

        // Wait for completion
        for try await _ in stream2.stream {}

        // Start generation in third conversation with long prompt (many tokens)
        let stream3 = await swiftyCore.start(
            prompt: "This is a very long and detailed prompt that contains many different words and concepts. It discusses various topics including technology, science, philosophy, and the nature of human consciousness. We are exploring the depths of language understanding and generation capabilities.",
            params: params,
            conversationId: conversationId3
        )

        // Wait for completion
        for try await _ in stream3.stream {}

        // Verify all conversations exist and have content
        let info1 = swiftyCore.getConversationInfo(conversationId1)
        let info2 = swiftyCore.getConversationInfo(conversationId2)
        let info3 = swiftyCore.getConversationInfo(conversationId3)

        #expect(info1 != nil, "First conversation should exist")
        #expect(info2 != nil, "Second conversation should exist")
        #expect(info3 != nil, "Third conversation should exist")

        #expect(info1?.messageCount == 2, "First conversation should have 2 messages")
        #expect(info2?.messageCount == 2, "Second conversation should have 2 messages")
        #expect(info3?.messageCount == 2, "Third conversation should have 2 messages")

        // Test that conversations maintain separate state by continuing each one
        // and verifying they don't interfere with each other

        // Continue first conversation with short prompt
        try swiftyCore.continueConversation(conversationId1)
        let continueStream1 = await swiftyCore.start(
            prompt: "Tell me more",
            params: params,
            conversationId: conversationId1
        )

        var response1 = ""
        for try await token in continueStream1.stream {
            response1 += token
        }

        // Continue second conversation with medium prompt
        try swiftyCore.continueConversation(conversationId2)
        let continueStream2 = await swiftyCore.start(
            prompt: "Can you elaborate on that topic and provide some additional details?",
            params: params,
            conversationId: conversationId2
        )

        var response2 = ""
        for try await token in continueStream2.stream {
            response2 += token
        }

        // Continue third conversation with long prompt
        try swiftyCore.continueConversation(conversationId3)
        let continueStream3 = await swiftyCore.start(
            prompt: "This is a continuation request that asks for detailed analysis and thorough explanation of the previous discussion points, including technical aspects, theoretical frameworks, and practical implications.",
            params: params,
            conversationId: conversationId3
        )

        var response3 = ""
        for try await token in continueStream3.stream {
            response3 += token
        }

        // Verify all conversations generated responses
        #expect(!response1.isEmpty, "First conversation should generate response")
        #expect(!response2.isEmpty, "Second conversation should generate response")
        #expect(!response3.isEmpty, "Third conversation should generate response")

        // Verify conversation state isolation by checking message counts
        let finalInfo1 = swiftyCore.getConversationInfo(conversationId1)
        let finalInfo2 = swiftyCore.getConversationInfo(conversationId2)
        let finalInfo3 = swiftyCore.getConversationInfo(conversationId3)

        #expect(finalInfo1?.messageCount == 4, "First conversation should have 4 messages (2 pairs)")
        #expect(finalInfo2?.messageCount == 4, "Second conversation should have 4 messages (2 pairs)")
        #expect(finalInfo3?.messageCount == 4, "Third conversation should have 4 messages (2 pairs)")

        // Test that switching between conversations doesn't cause state pollution
        // by starting a new generation in each conversation and verifying
        // they maintain their separate contexts

        // Switch to conversation 1 and generate with short prompt
        try swiftyCore.continueConversation(conversationId1)
        let finalStream1 = await swiftyCore.start(
            prompt: "End",
            params: params,
            conversationId: conversationId1
        )

        for try await _ in finalStream1.stream {}

        // Switch to conversation 2 and generate with medium prompt
        try swiftyCore.continueConversation(conversationId2)
        let finalStream2 = await swiftyCore.start(
            prompt: "Please provide a summary of our discussion so far",
            params: params,
            conversationId: conversationId2
        )

        for try await _ in finalStream2.stream {}

        // Switch to conversation 3 and generate with long prompt
        try swiftyCore.continueConversation(conversationId3)
        let finalStream3 = await swiftyCore.start(
            prompt: "Based on our extensive conversation covering multiple topics and concepts, please provide a detailed analysis that synthesizes all the key points, theoretical frameworks, and practical implications we've discussed.",
            params: params,
            conversationId: conversationId3
        )

        for try await _ in finalStream3.stream {}

        // Verify all conversations still exist and have grown
        let veryFinalInfo1 = swiftyCore.getConversationInfo(conversationId1)
        let veryFinalInfo2 = swiftyCore.getConversationInfo(conversationId2)
        let veryFinalInfo3 = swiftyCore.getConversationInfo(conversationId3)

        #expect(veryFinalInfo1?.messageCount == 6, "First conversation should have 6 messages")
        #expect(veryFinalInfo2?.messageCount == 6, "Second conversation should have 6 messages")
        #expect(veryFinalInfo3?.messageCount == 6, "Third conversation should have 6 messages")

        // Verify that conversations have different token counts by checking their content
        let conversations = swiftyCore.getAllConversations()
        #expect(conversations.count >= 3, "Should have at least 3 conversations")

        // Get the three test conversations
        let conv1 = conversations.first { $0.id == conversationId1 }
        let conv2 = conversations.first { $0.id == conversationId2 }
        let conv3 = conversations.first { $0.id == conversationId3 }

        #expect(conv1 != nil, "First conversation should exist")
        #expect(conv2 != nil, "Second conversation should exist")
        #expect(conv3 != nil, "Third conversation should exist")

        // Verify that conversations have different content lengths (indicating different token counts)
        let totalTokens1 = conv1!.messages.flatMap(\.tokens).count
        let totalTokens2 = conv2!.messages.flatMap(\.tokens).count
        let totalTokens3 = conv3!.messages.flatMap(\.tokens).count

        #expect(totalTokens1 > 0, "First conversation should have tokens")
        #expect(totalTokens2 > 0, "Second conversation should have tokens")
        #expect(totalTokens3 > 0, "Third conversation should have tokens")

        // Verify that conversations have different token counts (they should due to different prompt lengths)
        #expect(totalTokens1 != totalTokens2, "Conversations 1 and 2 should have different token counts")
        #expect(totalTokens2 != totalTokens3, "Conversations 2 and 3 should have different token counts")
        #expect(totalTokens1 != totalTokens3, "Conversations 1 and 3 should have different token counts")

        // Verify that the longest prompt (conv3) has the most tokens
        #expect(totalTokens3 > totalTokens2, "Third conversation should have more tokens than second")
        #expect(totalTokens2 > totalTokens1, "Second conversation should have more tokens than first")

        // Clean up
        await swiftyCore.cancelAll()
    }

    @Test("SwiftyCoreLlama conversation title functionality test")
    func conversationTitleFunctionalityTest() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for conversation title functionality test"
        )

        let swiftyCore = try SwiftyLlamaCore(modelPath: TestUtilities.testModelPath, contextSize: 512)

        let params = GenerationParams(temperature: 0.7, maxTokens: 10)

        // Test 1: Set a title manually
        let conversationId = swiftyCore.startNewConversation()

        do {
            try swiftyCore.setConversationTitle(conversationId, title: "Test Conversation")
            let retrievedTitle = swiftyCore.getConversationTitle(conversationId)
            #expect(retrievedTitle == "Test Conversation", "Should retrieve the set title")
        } catch {
            #expect(Bool(false), "Setting title should not throw an error")
        }

        // Test 2: Try to set a title that's too long
        do {
            let longTitle = String(repeating: "A", count: 201) // 201 characters
            try swiftyCore.setConversationTitle(conversationId, title: longTitle)
            #expect(Bool(false), "Setting title longer than 200 characters should throw an error")
        } catch let error as GenerationError {
            if case .titleTooLong = error {
                // This is expected
            } else {
                throw error
            }
        }

        // Test 3: Try to generate title for conversation with insufficient tokens
        do {
            try await swiftyCore.generateConversationTitle(conversationId)
            #expect(Bool(false), "Generating title for conversation with insufficient tokens should throw an error")
        } catch let error as GenerationError {
            if case .insufficientTokensForTitleGeneration = error {
                // This is expected
            } else {
                throw error
            }
        }

        // Test 4: Generate a conversation with enough content and then generate title
        let prompt = "Hello, I would like to discuss artificial intelligence and machine learning. Can you tell me about the differences between supervised and unsupervised learning?"

        let stream = await swiftyCore.start(prompt: prompt, params: params, conversationId: conversationId)

        // Consume the stream to generate a response
        var response = ""
        for try await token in stream.stream {
            response += token
        }

        // Verify the conversation has content
        let conversations = swiftyCore.getAllConversations()
        let conversation = conversations.first!
        #expect(!conversation.messages.isEmpty, "Conversation should have messages")
        #expect(conversation.totalTokens > 0, "Conversation should have tokens")
        print("Conversation has \(conversation.totalTokens) tokens")

        // Test 5: Generate title for conversation with sufficient content
        // First, let's add more content to the conversation to give the model more context
        let continuationPrompt = "Can you also explain deep learning and neural networks?"
        do {
            let continueStream = await swiftyCore.start(
                prompt: continuationPrompt,
                params: params,
                conversationId: conversationId
            )
            var continuationResponse = ""
            for try await token in continueStream.stream {
                continuationResponse += token
            }
            #expect(!continuationResponse.isEmpty, "Continuation should generate a response")
        } catch {
            print("Error continuing conversation: \(error)")
        }

        // Now try to generate a title with more content
        // Note: The TinyStories model may not generate meaningful titles, so we test the API functionality
        do {
            try await swiftyCore.generateConversationTitle(conversationId)
            let generatedTitle = swiftyCore.getConversationTitle(conversationId)
            print("Generated title: '\(generatedTitle ?? "nil")'")
            // The model might generate just punctuation, which gets cleaned to empty string
            // This is acceptable for this test model - we're testing the API, not the model quality
            if let title = generatedTitle, !title.isEmpty {
                #expect(title.count <= 200, "Generated title should not exceed 200 characters")
            } else {
                print("Model generated empty title (acceptable for TinyStories model)")
            }
        } catch {
            print("Title generation failed with error: \(error)")
            #expect(Bool(false), "Generating title for conversation with sufficient content should not throw an error")
        }

        // Test 6: Test auto-generation during save
        let newConversationId = swiftyCore.startNewConversation()
        let newPrompt = "Let's talk about renewable energy sources and their impact on climate change."

        let newStream = await swiftyCore.start(prompt: newPrompt, params: params, conversationId: newConversationId)

        // Consume the stream
        var newResponse = ""
        for try await token in newStream.stream {
            newResponse += token
        }

        // Verify the new conversation has no title initially
        let newConversations = swiftyCore.getAllConversations()
        let newConversation = newConversations.first { $0.id == newConversationId }!
        #expect(newConversation.title == nil, "New conversation should not have a title initially")

        // Save conversations (this should auto-generate titles)
        let jsonData = try await swiftyCore.exportConversations()
        #expect(!jsonData.isEmpty, "Should generate JSON data")

        // Verify that the new conversation now has a title
        let savedConversations = swiftyCore.getAllConversations()
        let savedNewConversation = savedConversations.first { $0.id == newConversationId }!
        print("Saved conversation title: \(savedNewConversation.title ?? "nil")")
        // The model might generate just punctuation, which gets cleaned to empty string
        // This is acceptable for this test model - we're testing the API, not the model quality
        if let title = savedNewConversation.title, !title.isEmpty {
            #expect(title.count <= 200, "Auto-generated title should not exceed 200 characters")
        } else {
            print("Model generated empty title for saved conversation (acceptable for TinyStories model)")
        }

        // Clean up
        await swiftyCore.cancelAll()
    }

    @Test("SwiftyCoreLlama context truncation validation test")
    func contextTruncationValidationTest() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for context truncation validation test"
        )

        let swiftyCore = try SwiftyLlamaCore(
            modelPath: TestUtilities.testModelPath,
            contextSize: 100
        ) // Small context size to force truncation

        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Create a conversation with a short history
        let shortHistoryPrompt = "Hello, this is a short message."

        let stream = await swiftyCore.start(prompt: shortHistoryPrompt, params: params, conversationId: nil)

        // Consume the stream to generate a response
        var response = ""
        for try await token in stream.stream {
            response += token
        }

        // Verify the conversation was created and has content
        let conversations = swiftyCore.getAllConversations()
        #expect(conversations.count == 1, "Should have exactly one conversation")

        let conversation = conversations.first!
        #expect(!conversation.messages.isEmpty, "Conversation should have messages")

        // Verify that the conversation has tokens
        let totalTokens = conversation.messages.flatMap(\.tokens).count
        #expect(totalTokens > 0, "Conversation should have tokens")

        // Now try to continue the conversation with a longer prompt that should trigger truncation
        let continuationPrompt = String(
            repeating: "This is a very long continuation message that will consume many tokens when tokenized. ",
            count: 50
        )

        // This should fail because our validation logic prevents context from exceeding the limit
        do {
            let continueStream = await swiftyCore.start(
                prompt: continuationPrompt,
                params: params,
                conversationId: conversation.id
            )

            // If we get here, the truncation logic worked
            var continuationResponse = ""
            for try await token in continueStream.stream {
                continuationResponse += token
            }

            // Verify the continuation worked
            #expect(!continuationResponse.isEmpty, "Continuation should generate a response")

            // Verify the conversation grew
            let updatedConversations = swiftyCore.getAllConversations()
            let updatedConversation = updatedConversations.first!
            #expect(updatedConversations.count == 1, "Should still have exactly one conversation")
            #expect(updatedConversation.messages.count > conversation.messages.count, "Conversation should have grown")

            // Verify that the total tokens don't exceed our small context size
            let updatedTotalTokens = updatedConversation.messages.flatMap(\.tokens).count
            #expect(updatedTotalTokens <= 100, "Total tokens should not exceed context size after truncation")

        } catch let error as GenerationError {
            // If context preparation fails, that's also acceptable - it means our validation is working
            if case .contextPreparationFailed = error {
                // This is expected behavior when validation prevents context overflow
                print("Context preparation failed as expected due to validation: \(error)")
            } else {
                throw error
            }
        }

        // Clean up
        await swiftyCore.cancelAll()
    }
}
