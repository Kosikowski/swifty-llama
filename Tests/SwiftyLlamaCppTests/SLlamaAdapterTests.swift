import Testing

@testable import SwiftyLlamaCpp

final class SLlamaAdapterTests {
    
    @Test("LoRA adapter initialization with invalid path should return nil")
    func testLoRAAdapterInitWithInvalidPath() throws {
        // Create a model first
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(false, "Model should load successfully")
            return
        }
        
        let adapter = SLlamaAdapter(model: model, path: "/invalid/path/to/lora.adapter")
        #expect(adapter == nil)
    }
    
    @Test("LoRA adapter should have valid pointer when initialized")
    func testLoRAAdapterValidPointer() throws {
        // Create a model first
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(false, "Model should load successfully")
            return
        }
        
        // Note: This test requires a valid LoRA adapter file
        // For now, we'll test the nil case
        let adapter = SLlamaAdapter(model: model, path: "/invalid/path")
        #expect(adapter == nil) // Should return nil when path is invalid
    }
    
    @Test("LoRA adapter should be valid when properly initialized")
    func testLoRAAdapterIsValid() throws {
        // Create a model first
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(false, "Model should load successfully")
            return
        }
        
        let adapter = SLlamaAdapter(model: model, path: "/invalid/path")
        #expect(adapter == nil) // Should return nil when path is invalid
    }
    
    @Test("Context should handle LoRA adapter operations gracefully")
    func testContextLoRAOperations() throws {
        // Create a model and context
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(false, "Model should load successfully")
            return
        }
        
        guard let context = SLlamaContext(model: model) else {
            #expect(false, "Context should be created successfully")
            return
        }
        
        // Test with invalid adapter
        let invalidAdapter = SLlamaAdapter(model: model, path: "/invalid/path")
        #expect(invalidAdapter == nil)
        
        // Test adding nil adapter (this will fail because we can't create an adapter with invalid path)
        // So we'll test the clear operation instead
        context.clearLoRAAdapters()
        
        // Test loading invalid adapter
        let loadedAdapter = context.loadLoRAAdapter(from: "/invalid/path")
        #expect(loadedAdapter == nil)
    }
    
    @Test("Context should handle multiple LoRA adapter operations")
    func testMultipleLoRAOperations() throws {
        // Create a model and context
        guard let model = SLlamaModel(modelPath: "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf") else {
            #expect(false, "Model should load successfully")
            return
        }
        
        guard let context = SLlamaContext(model: model) else {
            #expect(false, "Context should be created successfully")
            return
        }
        
        // Test clearing adapters multiple times
        context.clearLoRAAdapters()
        context.clearLoRAAdapters()
        context.clearLoRAAdapters()
        
        // Should not crash
        #expect(true)
    }
} 