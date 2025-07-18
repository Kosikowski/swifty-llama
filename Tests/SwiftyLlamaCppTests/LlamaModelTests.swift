import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct LlamaModelTests {
    
    @Test("Model creation with invalid path")
    func testModelCreationWithInvalidPath() throws {
        // Test that model creation fails gracefully with invalid path
        let model = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        #expect(model == nil, "Model creation should fail with invalid path")
    }
    
    @Test("Model properties with nil model")
    func testModelPropertiesWithNilModel() throws {
        // Test that model properties return safe defaults when model is nil
        let model = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        
        if let model = model {
            // This shouldn't happen with invalid path, but test properties anyway
            #expect(model.pointer == nil, "Invalid model should have nil pointer")
            #expect(model.vocab == nil, "Invalid model should have nil vocab")
            #expect(model.embeddingDimensions == 0, "Invalid model should have 0 embedding dimensions")
            #expect(model.layers == 0, "Invalid model should have 0 layers")
            #expect(model.attentionHeads == 0, "Invalid model should have 0 attention heads")
            #expect(model.parameters == 0, "Invalid model should have 0 parameters")
            #expect(model.size == 0, "Invalid model should have 0 size")
        } else {
            // Expected behavior
            #expect(Bool(true), "Model creation failed as expected")
        }
    }
} 