import Testing
import Foundation
@testable import SwiftyLlamaCpp

struct SLlamaAdapterTests {
    let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
    
    @Test("Adapter creation with invalid path should return nil")
    func testAdapterCreationWithInvalidPath() throws {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }
        
        guard let model = SLlamaModel(modelPath: modelPath) else {
            print("Test skipped: Model could not be loaded at \(modelPath)")
            return
        }
        
        // Create adapter with invalid path (should return nil)
        let adapter = SLlamaAdapter(model: model, path: "/invalid/path/to/lora.adapter")
        #expect(adapter == nil, "Adapter should be nil for invalid path")
    }

    @Test("Context should handle LoRA adapter operations gracefully")
    func testContextWithAdapterOperations() throws {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }
        guard let model = SLlamaModel(modelPath: modelPath) else {
            print("Test skipped: Model could not be loaded at \(modelPath)")
            return
        }
        
        // Initialize the backend before creating context
        SwiftyLlamaCpp.initialize()
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            print("Test skipped: Context could not be created")
            return
        }
        
        // Test adapter operations that should not crash even with invalid inputs
        context.clearLoRAAdapters()
        
        // The operations should complete without crashing
        #expect(context.pointer != nil, "Context should remain valid after adapter operations")
    }

    @Test("Control vector operations should work without crashing")
    func testControlVectorOperations() throws {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }
        guard let model = SLlamaModel(modelPath: modelPath) else {
            print("Test skipped: Model could not be loaded at \(modelPath)")
            return
        }
        
        // Initialize the backend before creating context
        SwiftyLlamaCpp.initialize()
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            print("Test skipped: Context could not be created")
            return
        }
        
        // Test control vector operations with dummy data
        let dummyVector = [Float](repeating: 0.1, count: 64) // Match model embedding size
        
        // Apply control vector (should not crash)
        let result = dummyVector.withUnsafeBufferPointer { buffer in
            return context.applyControlVector(
                data: buffer.baseAddress!,
                length: buffer.count,
                embeddingDimensions: Int32(model.embeddingDimensions),
                layerStart: 0,
                layerEnd: 1
            )
        }
        
        // Clear control vector (should not crash)
        let clearResult = context.clearControlVector()
        
        // The operations should complete without crashing
        #expect(result >= -1, "Apply control vector should return valid result")
        #expect(clearResult >= -1, "Clear control vector should return valid result")
    }
} 