import Foundation
import Testing
@testable import SLlama

struct SLlamaAdapterTests {
    @Test("Adapter initialization with invalid path throws error")
    func sLlamaAdapterInitialization() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()

        let model = try SLlamaModel(modelPath: modelPath)

        // Create adapter with invalid path should throw
        #expect(throws: SLlamaError.self) {
            try SLlamaAdapter(model: model, path: "/invalid/path/to/lora.adapter")
        }
    }

    @Test("Context LoRA adapter operations")
    func sLlamaContextLoRAOperations() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let context = try SLlamaContext(model: model)

        // Test that loading non-existent adapter throws
        #expect(throws: SLlamaError.self) {
            try context.loadLoRAAdapter(from: "/invalid/path/to/lora.adapter")
        }

        // The context operations should complete without crashing
        // Clear any adapters (should not throw)
        context.clearLoRAAdapters()
    }

    @Test("Context control vector operations")
    func sLlamaContextControlVectorOperations() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let context = try SLlamaContext(model: model)

        // Create dummy control vector data
        let dummyVector: [Float] = Array(repeating: 0.1, count: 100)

        // Apply control vector (test that it doesn't crash)
        try dummyVector.withUnsafeBufferPointer { buffer in
            try context.applyControlVector(
                data: buffer.baseAddress!,
                length: buffer.count,
                embeddingDimensions: 10,
                layerStart: 0,
                layerEnd: 2
            )
        }

        // Clear control vector (should not throw)
        try context.clearControlVector()

        // The operations should complete without crashing
        #expect(Bool(true), "Control vector operations completed successfully")
    }

    @Test("Adapter creation with null model throws error")
    func sLlamaAdapterNullModel() throws {
        SLlama.initialize()
        defer { SLlama.cleanup() }

        // Test that creating adapter with null model pointer throws
        #expect(throws: SLlamaError.self) {
            try SLlamaAdapter(model: SLlamaModel(modelPointer: nil), path: "/some/path")
        }
    }
}
