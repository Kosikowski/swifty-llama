import Testing
import SwiftyLlamaCpp

@testable import SwiftyLlamaCpp

struct SLlamaModelTests {
    
    @Test("Model loading and basic properties")
    func testModelLoading() throws {
        // Initialize backend
        SwiftyLlamaCpp.initialize()
        
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
        guard let model = SLlamaModel(modelPath: modelPath) else {
            #expect(Bool(false), "Model should load successfully")
            return
        }
        
        // Test basic properties
        #expect(model.embeddingDimensions > 0, "Embedding dimensions should be positive")
        #expect(model.layers > 0, "Layers should be positive")
        #expect(model.attentionHeads > 0, "Attention heads should be positive")
        #expect(model.kvAttentionHeads > 0, "KV attention heads should be positive")
        #expect(model.parameters > 0, "Parameters should be positive")
        #expect(model.size > 0, "Size should be positive")
        #expect(model.trainingContextLength > 0, "Training context length should be positive")
        
        // Test model type properties
        // Note: This GPT-2 model has rope type -1, which is valid for some models
        #expect(model.ropeType.rawValue >= -1, "RoPE type should be valid")
        #expect(model.ropeFreqScaleTrain > 0, "RoPE frequency scale should be positive")
        #expect(model.slidingWindowAttention >= 0, "Sliding window attention should be non-negative")
        
        // Test encoder/decoder properties
        #expect(model.hasEncoder == false, "GPT-2 model should not have encoder")
        #expect(model.hasDecoder == true, "GPT-2 model should have decoder")
        #expect(model.isRecurrent == false, "GPT-2 model should not be recurrent")
        
        // Test decoder start token (GPT-2 might not have a specific decoder start token)
        // Note: Some models return -1 for decoder start token, which is valid
        #expect(model.decoderStartToken >= -1, "Decoder start token should be valid")
        
        // Test metadata
        #expect(model.metadataCount > 0, "Model should have metadata")
        
        // Test description
        if let description = model.description() {
            #expect(!description.isEmpty, "Model description should not be empty")
        }
        
        // Test chat template (this model might not have one)
        _ = model.chatTemplate(named: "default")
        // Note: This model might not have a chat template, so we don't assert on it
        
        // Cleanup backend
        SwiftyLlamaCpp.cleanup()
    }
    
    @Test("Model metadata access")
    func testModelMetadataAccess() throws {
        // Initialize backend
        SwiftyLlamaCpp.initialize()
        
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
        guard let model = SLlamaModel(modelPath: modelPath) else {
            #expect(Bool(false), "Model should load successfully")
            return
        }
        
        let metadataCount = model.metadataCount
        #expect(metadataCount > 0, "Model should have metadata entries")
        
        // Test accessing metadata by index
        for i in 0..<metadataCount {
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
        
        // Cleanup backend
        SwiftyLlamaCpp.cleanup()
    }
} 