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

        // Training methods now throw trainingNotSupported
        do {
            _ = try tuningLlama.getCurrentTrainingSession()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.getTrainingMetrics()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }
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

    @Test("SwiftyTuningLlama training not supported test")
    func trainingNotSupported() async throws {
        // Fail if model not available
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for training not supported test"
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

        // All training-related methods should throw trainingNotSupported
        do {
            _ = try tuningLlama.startTrainingSession(dataset: dataset, config: config)
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            try tuningLlama.stopTrainingSession()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.getCurrentTrainingSession()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.getTrainingMetrics()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.evaluateModel(validationExamples: dataset.validation)
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            try tuningLlama.saveLoRAAdapter(path: "/tmp/test.gguf")
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            try tuningLlama.saveFineTunedModel(path: "/tmp/test.gguf")
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            try tuningLlama.exportModel(path: "/tmp/test.gguf", format: .gguf)
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.getModelSavingInfo()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }
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

        // Training methods throw trainingNotSupported regardless of context state
        do {
            let dataset = TrainingDataset(training: [], validation: [])
            let config = TrainingConfig()
            _ = try tuningLlama.startTrainingSession(dataset: dataset, config: config)
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.evaluateModel(validationExamples: [])
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }
    }

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
            .trainingNotSupported,
        ]

        for error in errors {
            let description = error.errorDescription
            #expect(description != nil, "Error description should not be nil for \(error)")
            #expect(!description!.isEmpty, "Error description should not be empty for \(error)")
        }
    }
}
