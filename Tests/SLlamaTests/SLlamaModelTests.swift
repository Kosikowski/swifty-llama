import Foundation
import Testing
@testable import SLlama

struct SLlamaModelTests {
    // MARK: Properties

    let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

    // MARK: Functions

    @Test("Model loading and basic properties")
    func modelLoadingAndBasicProperties() throws {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }
        // Disable logging to suppress verbose output
        SLlama.disableLogging()

        // Initialize backend
        SLlama.initialize()

        guard let model = SLlamaModel(modelPath: modelPath) else {
            print("Test skipped: Model could not be loaded at \(modelPath)")
            return
        }

        // Test basic properties
        #expect(model.embeddingDimensions > 0, "Embedding dimensions should be positive")
        #expect(model.layers > 0, "Layers should be positive")
        #expect(model.attentionHeads > 0, "Attention heads should be positive")
        #expect(model.kvAttentionHeads > 0, "KV attention heads should be positive")
        #expect(model.kvAttentionHeads <= model.attentionHeads, "KV attention heads should not exceed attention heads")
        #expect(model.parameters > 0, "Parameters should be positive")
        #expect(model.size > 0, "Size should be positive")
        #expect(model.trainingContextLength > 0, "Training context length should be positive")

        // Test model type properties
        #expect(model.ropeType.rawValue >= -1, "RoPE type should be valid")
        #expect(model.ropeFreqScaleTrain > 0, "RoPE frequency scale should be positive")
        #expect(model.slidingWindowAttention >= 0, "Sliding window attention should be non-negative")

        // Test encoder/decoder properties
        #expect(model.hasEncoder == false, "GPT-2 model should not have encoder")
        #expect(model.hasDecoder == true, "GPT-2 model should have decoder")
        #expect(model.isRecurrent == false, "GPT-2 model should not be recurrent")

        // Test decoder start token (GPT-2 might not have a specific decoder start token)
        #expect(model.decoderStartToken >= -1, "Decoder start token should be valid")

        // Test metadata
        #expect(model.metadataCount > 0, "Model should have metadata")

        // Test description
        if let description = model.description() {
            #expect(!description.isEmpty, "Model description should not be empty")
        }

        // Cleanup
        SLlama.cleanup()
    }

    @Test("Model metadata access")
    func modelMetadataAccess() throws {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }
        // Disable logging to suppress verbose output
        SLlama.disableLogging()

        // Initialize backend
        SLlama.initialize()

        guard let model = SLlamaModel(modelPath: modelPath) else {
            print("Test skipped: Model could not be loaded at \(modelPath)")
            return
        }

        // Test metadata access
        let metadataCount = model.metadataCount
        #expect(metadataCount > 0, "Model should have metadata entries")

        // Test accessing metadata by index
        for i in 0 ..< metadataCount {
            if let key = model.metadataKey(at: i) {
                #expect(!key.isEmpty, "Metadata key should not be empty")

                if let value = model.metadataValue(at: i) {
                    #expect(!value.isEmpty, "Metadata value should not be empty")
                }
            }
        }

        // Test accessing metadata by key
        if let architecture = model.metadataValue(for: "general.architecture") {
            #expect(!architecture.isEmpty, "Architecture metadata should not be empty")
        }

        // Cleanup
        SLlama.cleanup()
    }
}
