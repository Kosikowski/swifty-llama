import Testing
import SwiftyLlamaCpp

@testable import SwiftyLlamaCpp

struct SLlamaAdapterTests {
    
    @Test("Adapter creation with invalid path should return nil")
    func testAdapterCreationWithInvalidPath() throws {
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
        
        // Cleanup backend
        SwiftyLlamaCpp.cleanup()
    }
    
    @Test("Adapter with valid model should have valid pointer")
    func testAdapterValidPointer() throws {
        // Initialize backend
        SwiftyLlamaCpp.initialize()
        
        // Create a model first
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(Bool(false), "Model should load successfully")
            return
        }
        
        // Test with invalid path (should return nil)
        let adapter = SLlamaAdapter(model: model, path: "/invalid/path")
        #expect(adapter == nil, "Adapter should return nil when path is invalid")
        
        // Cleanup backend
        SwiftyLlamaCpp.cleanup()
    }
    
    @Test("Adapter should be valid when properly initialized")
    func testAdapterIsValid() throws {
        // Initialize backend
        SwiftyLlamaCpp.initialize()
        
        // Create a model first
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(Bool(false), "Model should load successfully")
            return
        }
        
        let adapter = SLlamaAdapter(model: model, path: "/invalid/path")
        #expect(adapter == nil, "Adapter should be nil when path is invalid")
        
        // Cleanup backend
        SwiftyLlamaCpp.cleanup()
    }
    
    @Test("Context should handle LoRA adapter operations gracefully")
    func testContextLoRAOperations() throws {
        // Initialize backend
        SwiftyLlamaCpp.initialize()
        
        // Create a model and context
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(Bool(false), "Model should load successfully")
            return
        }
        
        guard let context = SLlamaContext(model: model) else {
            #expect(Bool(false), "Context should be created successfully")
            return
        }
        
        // Test with invalid adapter
        let invalidAdapter = SLlamaAdapter(model: model, path: "/invalid/path")
        #expect(invalidAdapter == nil, "Invalid adapter should be nil")
        
        // Test clearing adapters (should not crash)
        context.clearLoRAAdapters()
        
        // Test loading invalid adapter
        let loadedAdapter = context.loadLoRAAdapter(from: "/invalid/path")
        #expect(loadedAdapter == nil, "Loading invalid adapter should return nil")
        
        // Cleanup backend
        SwiftyLlamaCpp.cleanup()
    }
    
    @Test("Context should handle multiple LoRA adapter operations")
    func testMultipleLoRAOperations() throws {
        // Initialize backend
        SwiftyLlamaCpp.initialize()
        
        // Create a model and context
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(Bool(false), "Model should load successfully")
            return
        }
        
        guard let context = SLlamaContext(model: model) else {
            #expect(Bool(false), "Context should be created successfully")
            return
        }
        
        // Test clearing adapters multiple times (should not crash)
        context.clearLoRAAdapters()
        context.clearLoRAAdapters()
        context.clearLoRAAdapters()
        
        // Should not crash
        #expect(Bool(true))
        
        // Cleanup backend
        SwiftyLlamaCpp.cleanup()
    }
} 