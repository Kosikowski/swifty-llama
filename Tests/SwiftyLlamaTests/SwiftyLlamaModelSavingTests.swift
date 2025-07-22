import Foundation
import Testing
@testable import SwiftyLlama
@testable import TestUtilities

@Suite("SwiftyLlama Model Saving Tests")
@SwiftyLlamaActor
struct SwiftyLlamaModelSavingTests {
    var tuner: SwiftyLlamaTuning!

    @Test("Save LoRA adapter success test")
    mutating func saveLoRAAdapterSuccess() async throws {
        // Given: Load model and apply LoRA
        tuner = SwiftyLlamaTuner()
        try tuner.loadModel(path: "test-model.gguf")
        try tuner.applyLoRA(path: "test-adapter.gguf", scale: 1.0, metadata: nil)

        // When: Save LoRA adapter
        let outputPath = "test-output-lora.gguf"
        try tuner.saveLoRAAdapter(path: outputPath)

        // Then: Verify file was created
        #expect(FileManager.default.fileExists(atPath: outputPath))

        // Clean up
        try? FileManager.default.removeItem(atPath: outputPath)
    }

    @Test("Save LoRA adapter with metadata test")
    mutating func saveLoRAAdapterWithMetadata() async throws {
        // Given: Load model and apply LoRA
        tuner = SwiftyLlamaTuner()
        try tuner.loadModel(path: "test-model.gguf")
        try tuner.applyLoRA(path: "test-adapter.gguf", scale: 1.0, metadata: nil)

        // When: Save LoRA adapter with custom metadata
        let metadata = LoRAMetadata(rank: 8, alpha: 16.0)
        let outputPath = "test-output-lora-with-metadata.gguf"
        try tuner.saveLoRAAdapter(path: outputPath, metadata: metadata)

        // Then: Verify file was created
        #expect(FileManager.default.fileExists(atPath: outputPath))

        // Clean up
        try? FileManager.default.removeItem(atPath: outputPath)
    }

    @Test("Save LoRA adapter no LoRA applied test")
    mutating func saveLoRAAdapterNoLoRAApplied() async throws {
        // Given: Load model without applying LoRA
        tuner = SwiftyLlamaTuner()
        try tuner.loadModel(path: "test-model.gguf")

        // When/Then: Try to save LoRA adapter should fail
        do {
            try tuner.saveLoRAAdapter(path: "test-output.gguf")
            #expect(Bool(false), "Expected error when no LoRA is applied")
        } catch TuningError.noLoRAApplied {
            // Expected error
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }
    }

    @Test("Save LoRA adapter context not initialized test")
    mutating func saveLoRAAdapterContextNotInitialized() async throws {
        // Given: No model loaded
        tuner = SwiftyLlamaTuner()

        // When/Then: Try to save LoRA adapter should fail
        do {
            try tuner.saveLoRAAdapter(path: "test-output.gguf")
            #expect(Bool(false), "Expected error when context is not initialized")
        } catch TuningError.contextNotInitialized {
            // Expected error
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }
    }

    @Test("Save fine-tuned model success test")
    mutating func saveFineTunedModelSuccess() async throws {
        // Given: Load model and apply LoRA
        tuner = SwiftyLlamaTuner()
        try tuner.loadModel(path: "test-model.gguf")
        try tuner.applyLoRA(path: "test-adapter.gguf", scale: 1.0, metadata: nil)

        // When: Save fine-tuned model
        let outputPath = "test-output-model.gguf"
        try tuner.saveFineTunedModel(path: outputPath, format: .gguf, metadata: nil)

        // Then: Verify file was created
        #expect(FileManager.default.fileExists(atPath: outputPath))

        // Clean up
        try? FileManager.default.removeItem(atPath: outputPath)
    }

    @Test("Save fine-tuned model with metadata test")
    mutating func saveFineTunedModelWithMetadata() async throws {
        // Given: Load model and apply LoRA
        tuner = SwiftyLlamaTuner()
        try tuner.loadModel(path: "test-model.gguf")
        try tuner.applyLoRA(path: "test-adapter.gguf", scale: 1.0, metadata: nil)

        // When: Save fine-tuned model with metadata
        let metadata = ModelMetadata(
            name: "Test Fine-tuned Model",
            description: "A test fine-tuned model",
            version: "1.0.0",
            author: "Test Author",
            license: "MIT",
            tags: ["test", "fine-tuned"],
            parameters: 7_000_000_000,
            contextLength: 4096
        )
        let outputPath = "test-output-model-with-metadata.gguf"
        try tuner.saveFineTunedModel(path: outputPath, format: .gguf, metadata: metadata)

        // Then: Verify file was created
        #expect(FileManager.default.fileExists(atPath: outputPath))

        // Clean up
        try? FileManager.default.removeItem(atPath: outputPath)
    }

    @Test("Save fine-tuned model no base model test")
    mutating func saveFineTunedModelNoBaseModel() async throws {
        // Given: No model loaded
        tuner = SwiftyLlamaTuner()

        // When/Then: Try to save fine-tuned model should fail
        do {
            try tuner.saveFineTunedModel(path: "test-output.gguf", format: .gguf, metadata: nil)
            #expect(Bool(false), "Expected error when base model is not loaded")
        } catch TuningError.baseModelNotLoaded {
            // Expected error
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }
    }

    @Test("Export model success test")
    mutating func exportModelSuccess() async throws {
        // Given: Load model and apply LoRA
        tuner = SwiftyLlamaTuner()
        try tuner.loadModel(path: "test-model.gguf")
        try tuner.applyLoRA(path: "test-adapter.gguf", scale: 1.0, metadata: nil)

        // When: Export model
        let outputPath = "test-export-model.gguf"
        try tuner.exportModel(path: outputPath, format: .gguf, quantization: nil)

        // Then: Verify file was created
        #expect(FileManager.default.fileExists(atPath: outputPath))

        // Clean up
        try? FileManager.default.removeItem(atPath: outputPath)
    }

    @Test("Export model with quantization test")
    mutating func exportModelWithQuantization() async throws {
        // Given: Load model and apply LoRA
        tuner = SwiftyLlamaTuner()
        try tuner.loadModel(path: "test-model.gguf")
        try tuner.applyLoRA(path: "test-adapter.gguf", scale: 1.0, metadata: nil)

        // When: Export model with quantization
        let quantization = ModelQuantization(
            type: .q4_0,
            bits: 4,
            groupSize: 32
        )
        let outputPath = "test-export-model-quantized.gguf"
        try tuner.exportModel(path: outputPath, format: .gguf, quantization: quantization)

        // Then: Verify file was created
        #expect(FileManager.default.fileExists(atPath: outputPath))

        // Clean up
        try? FileManager.default.removeItem(atPath: outputPath)
    }

    @Test("Get model saving info with model test")
    mutating func getModelSavingInfoWithModel() async throws {
        // Given: Load model and apply LoRA
        tuner = SwiftyLlamaTuner()
        try tuner.loadModel(path: "test-model.gguf")
        try tuner.applyLoRA(path: "test-adapter.gguf", scale: 1.0, metadata: nil)

        // When: Get model saving info
        let info = tuner.getModelSavingInfo()

        // Then: Verify info is correct
        #expect(info.baseModelPath == "test-model.gguf")
        #expect(info.currentLoRA != nil)
        #expect(info.currentLoRA?.path == "test-adapter.gguf")
        #expect(info.currentLoRA?.scale == 1.0)
    }

    @Test("Get model saving info without model test")
    mutating func getModelSavingInfoWithoutModel() async throws {
        // Given: No model loaded
        tuner = SwiftyLlamaTuner()

        // When: Get model saving info
        let info = tuner.getModelSavingInfo()

        // Then: Verify info is empty
        #expect(info.baseModelPath == nil)
        #expect(info.currentLoRA == nil)
        #expect(info.trainingConfig == nil)
        #expect(info.trainingMetrics == nil)
        #expect(info.trainingDataset == nil)
    }

    @Test("Complete model saving workflow test")
    mutating func completeModelSavingWorkflow() async throws {
        // Given: Complete setup with model, LoRA, and training
        tuner = SwiftyLlamaTuner()
        try tuner.loadModel(path: "test-model.gguf")
        try tuner.applyLoRA(path: "test-adapter.gguf", scale: 1.0, metadata: nil)

        let conversations = [
            TrainingConversation(
                id: "conv1",
                messages: [
                    TrainingMessage(role: .system, content: "You are helpful."),
                    TrainingMessage(role: .user, content: "What is 2+2?"),
                    TrainingMessage(role: .assistant, content: "2+2 equals 4."),
                ]
            ),
        ]

        let dataset = try tuner.prepareTrainingData(conversations: conversations, validationSplit: 0.2)
        let config = TrainingConfig(loraRank: 8, learningRate: 2e-5, epochs: 3)
        _ = try tuner.startTrainingSession(dataset: dataset, config: config)

        // When: Save LoRA adapter
        let loraPath = "test-saved-lora.gguf"
        try tuner.saveLoRAAdapter(path: loraPath)

        // And: Save fine-tuned model
        let modelPath = "test-saved-model.gguf"
        let metadata = ModelMetadata(
            name: "Test Model",
            description: "A test fine-tuned model",
            version: "1.0.0"
        )
        try tuner.saveFineTunedModel(path: modelPath, format: .gguf, metadata: metadata)

        // And: Export quantized model
        let exportPath = "test-exported-model.gguf"
        let quantization = ModelQuantization(type: .q4_0, bits: 4)
        try tuner.exportModel(path: exportPath, format: .gguf, quantization: quantization)

        // Then: Verify all files were created
        #expect(FileManager.default.fileExists(atPath: loraPath))
        #expect(FileManager.default.fileExists(atPath: modelPath))
        #expect(FileManager.default.fileExists(atPath: exportPath))

        // And: Verify model saving info is complete
        let info = tuner.getModelSavingInfo()
        #expect(info.baseModelPath != nil)
        #expect(info.currentLoRA != nil)
        #expect(info.trainingConfig != nil)
        #expect(info.trainingMetrics != nil)
        #expect(info.trainingDataset != nil)

        // Clean up
        try? FileManager.default.removeItem(atPath: loraPath)
        try? FileManager.default.removeItem(atPath: modelPath)
        try? FileManager.default.removeItem(atPath: exportPath)
    }
}
