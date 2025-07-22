import Foundation
import Testing
@testable import SwiftyLlama
@testable import TestUtilities

@Suite("SwiftyTuningLlama Tests")
@SwiftyLlamaActor
struct SwiftyTuningLlamaTests {
    @Test("SwiftyTuningLlama initialization test")
    func initialization() async throws {
        let tuningLlama = SwiftyTuningLlama()

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

        let tuningLlama = SwiftyTuningLlama()

        // Load model
        try tuningLlama.loadModel(path: TestUtilities.testModelPath)

        // Verify model is loaded
        #expect(tuningLlama.getCurrentLoRA() == nil) // No LoRA applied yet
        #expect(tuningLlama.getAvailableAdapters().isEmpty)
    }

    @Test("SwiftyTuningLlama LoRA adapter management test")
    func loRAAdapterManagement() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for LoRA adapter management test"
        )

        let tuningLlama = SwiftyTuningLlama()

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

        let tuningLlama = SwiftyTuningLlama()

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
        #expect(dataset.training.count + dataset.validation.count == 2)
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

        let tuningLlama = SwiftyTuningLlama()

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
        #expect(session.dataset.training.count + session.dataset.validation.count == 1)
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

        let tuningLlama = SwiftyTuningLlama()

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

        let tuningLlama = SwiftyTuningLlama()

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

        let dataset = try tuningLlama.prepareTrainingData(conversations: conversations)

        // Evaluate model
        let metrics = try tuningLlama.evaluateModel(validationExamples: dataset.validation)

        // Verify metrics structure
        #expect(metrics.totalExamples == dataset.validation.count)
        #expect(metrics.totalTokens > 0)
        #expect(metrics.perplexity >= 0)
        #expect(metrics.averageLoss >= 0)
    }

    @Test("SwiftyTuningLlama LoRA compatibility test")
    func loRACompatibility() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for LoRA compatibility test"
        )

        let tuningLlama = SwiftyTuningLlama()

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
    func loRAFallbackMode() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for LoRA fallback mode test"
        )

        let tuningLlama = SwiftyTuningLlama()

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
        let tuningLlama = SwiftyTuningLlama()

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

        let tuningLlama = SwiftyTuningLlama()

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
}
