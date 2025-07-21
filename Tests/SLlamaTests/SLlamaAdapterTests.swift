import Foundation
import Testing
import TestUtilities
@testable import SLlama

struct SLlamaAdapterTests {
    @Test("Adapter initialization with invalid path throws error")
    func sLlamaAdapterInitialization() throws {
        let modelPath = TestUtilities.testModelPath

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

    @Test("Context LoRA adapter operations")
    func sLlamaContextLoRAOperations() throws {
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let context = try SLlamaContext(model: model)

        // Test basic LoRA operations (without actual adapter files)
        // Clear adapters (should not throw even if none loaded)
        context.clearLoRAAdapters()

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
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let context = try SLlamaContext(model: model)

        // Get the actual embedding dimensions from the model
        let embeddingDim = model.embeddingDimensions
        guard embeddingDim > 0 else {
            print("Test skipped: Model has invalid embedding dimensions: \(embeddingDim)")
            return
        }

        // Create control vector data with correct size (embeddingDim per layer)
        let numLayers = 2 // layerEnd - layerStart
        let vectorSize = Int(embeddingDim * Int32(numLayers))
        let dummyVector: [Float] = Array(repeating: 0.1, count: vectorSize)

        // Apply control vector (test that it doesn't crash)
        try dummyVector.withUnsafeBufferPointer { buffer in
            try context.applyControlVector(
                data: buffer.baseAddress!,
                length: buffer.count,
                embeddingDimensions: embeddingDim,
                layerStart: 0,
                layerEnd: 2
            )
        }

        // Clear control vector (should not throw)
        try context.clearControlVector()

        // Test that operations complete without crashing
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
