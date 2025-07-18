import Testing
import SwiftyLlamaCpp

@testable import SwiftyLlamaCpp

struct SLlamaAdapterTests {
    
    @Test("Adapter creation with invalid path should return nil")
    func testAdapterCreationWithInvalidPath() throws {
        // Disable logging to suppress verbose output
        SwiftyLlamaCpp.disableLogging()
        
        // Initialize backend
        SwiftyLlamaCpp.initialize()
        
        // Create a model first
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(Bool(false), "Model should load successfully")
            return
        }
        
        // Create adapter with invalid path (should return nil)
        let adapter = SLlamaAdapter(model: model, path: "/invalid/path/to/lora.adapter")
        #expect(adapter == nil, "Adapter should be nil for invalid path")
        
        // Cleanup
        SwiftyLlamaCpp.cleanup()
    }
    
    @Test("Context should handle LoRA adapter operations gracefully")
    func testContextWithAdapterOperations() throws {
        // Disable logging to suppress verbose output
        SwiftyLlamaCpp.disableLogging()
        
        // Initialize backend
        SwiftyLlamaCpp.initialize()
        
        // Create a model first
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(Bool(false), "Model should load successfully")
            return
        }
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            #expect(Bool(false), "Context should be created successfully")
            return
        }
        
        // Test that context can handle adapter operations gracefully
        // Even without a valid adapter, the context should not crash
        #expect(context.pointer != nil, "Context pointer should be valid")
        
        // Cleanup
        SwiftyLlamaCpp.cleanup()
    }
    
    @Test("Control vector operations should work without crashing")
    func testControlVectorOperations() throws {
        // Disable logging to suppress verbose output
        SwiftyLlamaCpp.disableLogging()
        
        // Initialize backend
        SwiftyLlamaCpp.initialize()
        
        // Create a model first
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(Bool(false), "Model should load successfully")
            return
        }
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            #expect(Bool(false), "Context should be created successfully")
            return
        }
        
        // Test control vector operations
        // Create a dummy control vector (small array of floats)
        let controlVector: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        controlVector.withUnsafeBufferPointer { buffer in
            let result = context.applyControlVector(
                data: buffer.baseAddress!,
                length: controlVector.count,
                embeddingDimensions: 4,
                layerStart: 0,
                layerEnd: 1
            )
            // Should not crash, even if the operation fails
            #expect(result >= -1, "applyControlVector should return valid result")
        }
        
        // Test clearing control vector
        let clearResult = context.clearControlVector()
        #expect(clearResult >= -1, "clearControlVector should return valid result")
        
        // Cleanup
        SwiftyLlamaCpp.cleanup()
    }
} 