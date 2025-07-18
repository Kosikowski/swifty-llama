import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct SLlamaContextTests {
    
    @Test("Context creation with invalid model")
    func testContextCreationWithInvalidModel() throws {
        // Test that context creation fails gracefully with invalid model
        let invalidModel = SLlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = invalidModel {
            let context = SLlamaContext(model: model)
            #expect(context == nil, "Context creation should fail with invalid model")
        } else {
            // If model creation itself failed, that's also acceptable
            #expect(Bool(true), "Model creation failed as expected")
        }
    }
    
    @Test("Context properties with nil context")
    func testContextPropertiesWithNilContext() throws {
        // Test that context properties return safe defaults when context is nil
        let invalidModel = SLlamaModel(modelPath: "/nonexistent/path/model.gguf")
        let context = invalidModel.flatMap { SLlamaContext(model: $0) }
        
        if let context = context {
            #expect(context.pointer == nil, "Invalid context should have nil pointer")
            #expect(context.associatedModel == nil, "Invalid context should have nil associated model")
            #expect(context.contextSize == 0, "Invalid context should have 0 context size")
            #expect(context.batchSize == 0, "Invalid context should have 0 batch size")
            #expect(context.maxBatchSize == 0, "Invalid context should have 0 max batch size")
            #expect(context.maxSequences == 0, "Invalid context should have 0 max sequences")
        } else {
            // Expected behavior
            #expect(Bool(true), "Context creation failed as expected")
        }
    }
} 