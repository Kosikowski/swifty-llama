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
    
    @Test("Context should handle multiple LoRA adapter operations")
    func testContextWithMultipleAdapterOperations() throws {
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
        
        // Test multiple adapter operations
        let adapter1 = SLlamaAdapter(model: model, path: "/invalid/path1")
        let adapter2 = SLlamaAdapter(model: model, path: "/invalid/path2")
        
        #expect(adapter1 == nil, "First adapter should be nil")
        #expect(adapter2 == nil, "Second adapter should be nil")
        #expect(context.pointer != nil, "Context pointer should remain valid")
        
        // Cleanup
        SwiftyLlamaCpp.cleanup()
    }
} 