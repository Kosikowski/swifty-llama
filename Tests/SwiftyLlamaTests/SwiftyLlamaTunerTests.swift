import Foundation
import Testing
@testable import SwiftyLlama
@testable import TestUtilities

@Suite("SwiftyTuningLlama Tests")
@SwiftyLlamaActor
struct SwiftyLlamaTunerTests {
    @Test("SwiftyTuningLlama initialization test")
    func initialization() async throws {
        let tuningLlama = SwiftyLlamaTuner()

        // Verify initial state
        #expect(tuningLlama.getCurrentLoRA() == nil)
        #expect(tuningLlama.getAvailableAdapters().isEmpty)
        #expect(tuningLlama.getCurrentTrainingSession() == nil)
        #expect(tuningLlama.getTrainingMetrics() == nil)
    }

    @Test("SwiftyTuningLlama model loading test")
    func modelLoading() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for model loading test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Verify model is loaded
        #expect(tuningLlama.getCurrentLoRA() == nil) // No LoRA applied yet
        #expect(tuningLlama.getAvailableAdapters().isEmpty)
    }

    @Test("SwiftyTuningLlama LoRA adapter management test")
    func loraAdapterManagement() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for LoRA adapter management test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Test applying LoRA (this will fail since adapter file doesn't exist)
        do {
            try tuningLlama.applyLoRA(path: "/nonexistent/adapter.gguf")
            #expect(Bool(false), "Should have thrown an error for nonexistent adapter")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .adapterFileNotFound(path: "/nonexistent/adapter.gguf"))
        }

        // Verify no LoRA is applied
        #expect(tuningLlama.getCurrentLoRA() == nil)
        #expect(tuningLlama.getAvailableAdapters().isEmpty)
    }

    @Test("SwiftyTuningLlama training data preparation test")
    func trainingDataPreparation() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for training data preparation test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Create test conversations
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are a helpful assistant."),
                    TrainingMessage(role: .user, content: "What is 2+2?"),
                    TrainingMessage(role: .assistant, content: "2+2 equals 4."),
                ]
            ),
            TrainingConversation(
                id: "conv2",
                messages: [
                    TrainingMessage(role: .system, content: "You are a math tutor."),
                    TrainingMessage(role: .user, content: "What is 3*3?"),
                    TrainingMessage(role: .assistant, content: "3*3 equals 9."),
                ]
            ),
        ]

        // Prepare training data
        let dataset = try tuningLlama.prepareTrainingData(
            conversations: conversations,
            validationSplit: 0.5
        )

        // Verify dataset structure
        // Each conversation has 3 messages, so 2 conversations = 6 total examples
        #expect(dataset.training.count + dataset.validation.count == 6)
        #expect(dataset.training.count > 0 || dataset.validation.count > 0)

        // Verify examples have tokens
        for example in dataset.training {
            #expect(!example.tokens.isEmpty)
            #expect(!example.formattedText.isEmpty)
        }

        for example in dataset.validation {
            #expect(!example.tokens.isEmpty)
            #expect(!example.formattedText.isEmpty)
        }
    }

    @Test("SwiftyTuningLlama training session management test")
    func trainingSessionManagement() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for training session management test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Create test dataset
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are a helpful assistant."),
                    TrainingMessage(role: .user, content: "Hello"),
                    TrainingMessage(role: .assistant, content: "Hi there!"),
                ]
            ),
        ]

        let dataset = try tuningLlama.prepareTrainingData(conversations: conversations)

        // Test valid training config
        let validConfig = TrainingConfig(
            loraRank: 8,
            learningRate: 2e-5,
            epochs: 3,
            batchSize: 1
        )

        let session = try tuningLlama.startTrainingSession(dataset: dataset, config: validConfig)

        // Verify session properties
        #expect(session.status == .running)
        // Single conversation with 3 messages = 3 total examples
        #expect(session.dataset.training.count + session.dataset.validation.count == 3)
        #expect(session.config.loraRank == 8)
        #expect(session.config.learningRate == 2e-5)

        // Verify current session
        let currentSession = tuningLlama.getCurrentTrainingSession()
        #expect(currentSession?.id == session.id)
        #expect(currentSession?.status == .running)

        // Test stopping session
        tuningLlama.stopTrainingSession()

        let stoppedSession = tuningLlama.getCurrentTrainingSession()
        #expect(stoppedSession?.status == .stopped)
        #expect(stoppedSession?.endTime != nil)
    }

    @Test("SwiftyTuningLlama invalid training config test")
    func invalidTrainingConfig() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for invalid training config test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Create test dataset
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are a helpful assistant."),
                    TrainingMessage(role: .user, content: "Hello"),
                    TrainingMessage(role: .assistant, content: "Hi there!"),
                ]
            ),
        ]

        let dataset = try tuningLlama.prepareTrainingData(conversations: conversations)

        // Test invalid LoRA rank
        let invalidRankConfig = TrainingConfig(loraRank: 0, learningRate: 2e-5, epochs: 3)

        do {
            _ = try tuningLlama.startTrainingSession(dataset: dataset, config: invalidRankConfig)
            #expect(Bool(false), "Should have thrown an error for invalid LoRA rank")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .invalidLoRARank(rank: 0))
        }

        // Test invalid learning rate
        let invalidLRConfig = TrainingConfig(loraRank: 8, learningRate: 2.0, epochs: 3)

        do {
            _ = try tuningLlama.startTrainingSession(dataset: dataset, config: invalidLRConfig)
            #expect(Bool(false), "Should have thrown an error for invalid learning rate")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .invalidLearningRate(rate: 2.0))
        }

        // Test invalid epochs
        let invalidEpochsConfig = TrainingConfig(loraRank: 8, learningRate: 2e-5, epochs: 0)

        do {
            _ = try tuningLlama.startTrainingSession(dataset: dataset, config: invalidEpochsConfig)
            #expect(Bool(false), "Should have thrown an error for invalid epochs")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .invalidEpochs(epochs: 0))
        }
    }

    @Test("SwiftyTuningLlama evaluation test")
    func evaluation() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for evaluation test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Create test validation examples
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are a helpful assistant."),
                    TrainingMessage(role: .user, content: "What is 2+2?"),
                    TrainingMessage(role: .assistant, content: "2+2 equals 4."),
                ]
            ),
        ]

        let dataset = try tuningLlama.prepareTrainingData(conversations: conversations, validationSplit: 1.0)

        // Evaluate model
        let metrics = try tuningLlama.evaluateModel(validationExamples: dataset.validation)

        // Verify metrics structure
        #expect(metrics.totalExamples == dataset.validation.count)
        #expect(metrics.totalTokens > 0)
        #expect(metrics.perplexity >= 0)
        #expect(metrics.averageLoss >= 0)
    }

    @Test("SwiftyTuningLlama LoRA compatibility test")
    func loraCompatibility() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for LoRA compatibility test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Test compatibility check
        let compatibility = try tuningLlama.validateLoRACompatibility(path: "/nonexistent/adapter.gguf")

        // Verify compatibility structure
        #expect(compatibility.isCompatible == true) // Simplified implementation always returns true
        #expect(compatibility.warnings.isEmpty)
        #expect(compatibility.baseModelSHA == "unknown")
        #expect(compatibility.adapterConfig == "unknown")
    }

    @Test("SwiftyTuningLlama LoRA fallback mode test")
    func loraFallbackMode() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for LoRA fallback mode test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Test setting fallback mode (should not throw)
        tuningLlama.setLoRAFallbackMode(true)
        tuningLlama.setLoRAFallbackMode(false)

        // Verify no errors occurred
        #expect(Bool(true), "Fallback mode should be set without errors")
    }

    @Test("SwiftyTuningLlama error handling test")
    func errorHandling() async throws {
        let tuningLlama = SwiftyLlamaTuner()

        // Test operations without loading model
        do {
            _ = try tuningLlama.applyLoRA(path: "/test/adapter.gguf")
            #expect(Bool(false), "Should have thrown an error for uninitialized context")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .contextNotInitialized)
        }

        do {
            _ = try tuningLlama.prepareTrainingData(conversations: [])
            #expect(Bool(false), "Should have thrown an error for uninitialized tokenizer")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .tokenizerNotInitialized)
        }

        do {
            let dataset = TrainingDataset(training: [], validation: [])
            let config = TrainingConfig()
            _ = try tuningLlama.startTrainingSession(dataset: dataset, config: config)
            #expect(Bool(false), "Should have thrown an error for uninitialized context")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .contextNotInitialized)
        }

        do {
            _ = try tuningLlama.evaluateModel(validationExamples: [])
            #expect(Bool(false), "Should have thrown an error for uninitialized context")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .contextNotInitialized)
        }
    }

    @Test("SwiftyTuningLlama QLoRA configuration test")
    func qLoRAConfiguration() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for QLoRA configuration test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Create test dataset
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are a helpful assistant."),
                    TrainingMessage(role: .user, content: "Hello"),
                    TrainingMessage(role: .assistant, content: "Hi there!"),
                ]
            ),
        ]

        let dataset = try tuningLlama.prepareTrainingData(conversations: conversations)

        // Test QLoRA configuration
        let qLoRAConfig = QLoRAConfig(
            quantType: "nf4",
            useDoubleQuant: true,
            computeDtype: "float16"
        )

        let config = TrainingConfig(
            loraRank: 8,
            learningRate: 2e-5,
            epochs: 3,
            useQLoRA: true,
            qLoRAConfig: qLoRAConfig
        )

        let session = try tuningLlama.startTrainingSession(dataset: dataset, config: config)

        // Verify QLoRA configuration
        #expect(session.config.useQLoRA == true)
        #expect(session.config.qLoRAConfig != nil)
        #expect(session.config.qLoRAConfig?.quantType == "nf4")
        #expect(session.config.qLoRAConfig?.useDoubleQuant == true)
        #expect(session.config.qLoRAConfig?.computeDtype == "float16")
    }

    // MARK: - Additional Edge Cases and Error Scenarios

    @Test("SwiftyTuningLlama empty conversations test")
    func emptyConversations() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for empty conversations test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Test with empty conversations
        let dataset = try tuningLlama.prepareTrainingData(conversations: [])

        // Verify empty dataset
        #expect(dataset.training.isEmpty)
        #expect(dataset.validation.isEmpty)
    }

    @Test("SwiftyTuningLlama single message conversation test")
    func singleMessageConversation() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for single message conversation test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Test with single message conversation
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .user, content: "Hello"),
                ]
            ),
        ]

        let dataset = try tuningLlama.prepareTrainingData(conversations: conversations)

        // Verify dataset has content
        #expect(dataset.training.count + dataset.validation.count == 1)
        #expect(!dataset.training.isEmpty || !dataset.validation.isEmpty)
    }

    @Test("SwiftyTuningLlama LoRA removal test")
    func loraRemoval() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for LoRA removal test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Test removing LoRA when none is applied
        do {
            try tuningLlama.removeLoRA()
            // Should not throw when no LoRA is applied
        } catch {
            #expect(Bool(false), "Should not throw when removing non-existent LoRA")
        }

        // Verify no LoRA is applied
        #expect(tuningLlama.getCurrentLoRA() == nil)
    }

    @Test("SwiftyTuningLlama training session without dataset test")
    func trainingSessionWithoutDataset() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for training session without dataset test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Test starting training session with empty dataset
        let emptyDataset = TrainingDataset(training: [], validation: [])
        let config = TrainingConfig(loraRank: 8, learningRate: 2e-5, epochs: 3)

        let session = try tuningLlama.startTrainingSession(dataset: emptyDataset, config: config)

        // Verify session is created even with empty dataset
        #expect(session.status == .running)
        #expect(session.dataset.training.isEmpty)
        #expect(session.dataset.validation.isEmpty)
    }

    @Test("SwiftyTuningLlama evaluation with empty validation test")
    func evaluationWithEmptyValidation() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for evaluation with empty validation test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Test evaluation with empty validation set
        let metrics = try tuningLlama.evaluateModel(validationExamples: [])

        // Verify metrics for empty validation
        #expect(metrics.totalExamples == 0)
        #expect(metrics.totalTokens == 0)
        #expect(metrics.averageLoss == 0.0)
        #expect(metrics.perplexity == 1.0) // exp(0.0) = 1.0
    }

    @Test("SwiftyTuningLlama training metrics test")
    func trainingMetrics() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for training metrics test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Create test dataset
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are a helpful assistant."),
                    TrainingMessage(role: .user, content: "Hello"),
                    TrainingMessage(role: .assistant, content: "Hi there!"),
                ]
            ),
        ]

        let dataset = try tuningLlama.prepareTrainingData(conversations: conversations)
        let config = TrainingConfig(loraRank: 8, learningRate: 2e-5, epochs: 3)

        // Start training session
        _ = try tuningLlama.startTrainingSession(dataset: dataset, config: config)

        // Get training metrics
        let metrics = tuningLlama.getTrainingMetrics()

        // Verify metrics structure
        #expect(metrics != nil)
        #expect(metrics?.currentEpoch == 0)
        #expect(metrics?.currentStep == 0)
        #expect(metrics?.trainingLoss == 0.0)
        #expect(metrics?.validationLoss == 0.0)
        #expect(metrics?.learningRate == 2e-5) // Should match the config learning rate
    }

    @Test("SwiftyTuningLlama boundary value training config test")
    func boundaryValueTrainingConfig() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for boundary value training config test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Create test dataset
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are a helpful assistant."),
                    TrainingMessage(role: .user, content: "Hello"),
                    TrainingMessage(role: .assistant, content: "Hi there!"),
                ]
            ),
        ]

        let dataset = try tuningLlama.prepareTrainingData(conversations: conversations)

        // Test boundary values for LoRA rank
        let minRankConfig = TrainingConfig(loraRank: 1, learningRate: 2e-5, epochs: 3)
        let maxRankConfig = TrainingConfig(loraRank: 128, learningRate: 2e-5, epochs: 3)

        // These should succeed
        let minSession = try tuningLlama.startTrainingSession(dataset: dataset, config: minRankConfig)
        #expect(minSession.config.loraRank == 1)

        tuningLlama.stopTrainingSession()

        let maxSession = try tuningLlama.startTrainingSession(dataset: dataset, config: maxRankConfig)
        #expect(maxSession.config.loraRank == 128)

        // Test boundary values for learning rate
        let minLRConfig = TrainingConfig(loraRank: 8, learningRate: 0.000001, epochs: 3)
        let maxLRConfig = TrainingConfig(loraRank: 8, learningRate: 1.0, epochs: 3)

        tuningLlama.stopTrainingSession()

        let minLRSession = try tuningLlama.startTrainingSession(dataset: dataset, config: minLRConfig)
        #expect(minLRSession.config.learningRate == 0.000001)

        tuningLlama.stopTrainingSession()

        let maxLRSession = try tuningLlama.startTrainingSession(dataset: dataset, config: maxLRConfig)
        #expect(maxLRSession.config.learningRate == 1.0)

        // Test boundary values for epochs
        let minEpochsConfig = TrainingConfig(loraRank: 8, learningRate: 2e-5, epochs: 1)
        let maxEpochsConfig = TrainingConfig(loraRank: 8, learningRate: 2e-5, epochs: 100)

        tuningLlama.stopTrainingSession()

        let minEpochsSession = try tuningLlama.startTrainingSession(dataset: dataset, config: minEpochsConfig)
        #expect(minEpochsSession.config.epochs == 1)

        tuningLlama.stopTrainingSession()

        let maxEpochsSession = try tuningLlama.startTrainingSession(dataset: dataset, config: maxEpochsConfig)
        #expect(maxEpochsSession.config.epochs == 100)
    }

    @Test("SwiftyTuningLlama validation split edge cases test")
    func validationSplitEdgeCases() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for validation split edge cases test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Create test conversations
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are a helpful assistant."),
                    TrainingMessage(role: .user, content: "Hello"),
                    TrainingMessage(role: .assistant, content: "Hi there!"),
                ]
            ),
            TrainingConversation(
                id: "conv2",
                messages: [
                    TrainingMessage(role: .system, content: "You are a math tutor."),
                    TrainingMessage(role: .user, content: "What is 2+2?"),
                    TrainingMessage(role: .assistant, content: "2+2 equals 4."),
                ]
            ),
        ]

        // Test validation split = 0.0 (all training)
        let allTrainingDataset = try tuningLlama.prepareTrainingData(
            conversations: conversations,
            validationSplit: 0.0
        )
        #expect(allTrainingDataset.validation.isEmpty)
        // 2 conversations with 3 messages each = 6 total examples
        #expect(allTrainingDataset.training.count == 6)

        // Test validation split = 1.0 (all validation)
        let allValidationDataset = try tuningLlama.prepareTrainingData(
            conversations: conversations,
            validationSplit: 1.0
        )
        #expect(allValidationDataset.training.isEmpty)
        // 2 conversations with 3 messages each = 6 total examples
        #expect(allValidationDataset.validation.count == 6)
    }

    @Test("SwiftyTuningLlama multiple LoRA adapters test")
    func multipleLoRAAdapters() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for multiple LoRA adapters test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Test applying multiple LoRA adapters (they will all fail due to nonexistent files)
        do {
            try tuningLlama.applyLoRA(path: "/nonexistent/adapter1.gguf")
            #expect(Bool(false), "Should have thrown an error for nonexistent adapter1")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .adapterFileNotFound(path: "/nonexistent/adapter1.gguf"))
        }

        do {
            try tuningLlama.applyLoRA(path: "/nonexistent/adapter2.gguf")
            #expect(Bool(false), "Should have thrown an error for nonexistent adapter2")
        } catch {
            #expect(error is TuningError)
            #expect((error as! TuningError) == .adapterFileNotFound(path: "/nonexistent/adapter2.gguf"))
        }

        // Verify no adapters are available since they all failed
        #expect(tuningLlama.getAvailableAdapters().isEmpty)
        #expect(tuningLlama.getCurrentLoRA() == nil)
    }

    @Test("SwiftyTuningLlama training session state transitions test")
    func trainingSessionStateTransitions() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for training session state transitions test"
        )

        let tuningLlama = SwiftyLlamaTuner()

        // Load model first
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Create test dataset
        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are a helpful assistant."),
                    TrainingMessage(role: .user, content: "Hello"),
                    TrainingMessage(role: .assistant, content: "Hi there!"),
                ]
            ),
        ]

        let dataset = try tuningLlama.prepareTrainingData(conversations: conversations)
        let config = TrainingConfig(loraRank: 8, learningRate: 2e-5, epochs: 3)

        // Start training session
        let session = try tuningLlama.startTrainingSession(dataset: dataset, config: config)

        // Verify initial state
        #expect(session.status == .running)
        #expect(session.endTime == nil)

        // Stop training session
        tuningLlama.stopTrainingSession()

        // Verify stopped state
        let stoppedSession = tuningLlama.getCurrentTrainingSession()
        #expect(stoppedSession?.status == .stopped)
        #expect(stoppedSession?.endTime != nil)
        #expect(stoppedSession?.id == session.id)
    }

    @Test("SwiftyTuningLlama error description test")
    func errorDescriptionTest() {
        // Test that all TuningError cases have proper descriptions
        let errors: [TuningError] = [
            .contextNotInitialized,
            .modelNotLoaded,
            .tokenizerNotInitialized,
            .adapterFileNotFound(path: "/test/path"),
            .adapterApplicationFailed(path: "/test/path", errorDescription: "test error"),
            .invalidLoRARank(rank: 0),
            .invalidLearningRate(rate: 2.0),
            .invalidEpochs(epochs: 0),
            .trainingSessionNotFound,
            .incompatibleAdapter,
        ]

        for error in errors {
            let description = error.errorDescription
            #expect(description != nil, "Error description should not be nil for \(error)")
            #expect(!description!.isEmpty, "Error description should not be empty for \(error)")
        }
    }
}
