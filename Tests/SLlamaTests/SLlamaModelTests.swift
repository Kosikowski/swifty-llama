import Foundation
import Testing
import TestUtilities
@testable import SLlama

struct SLlamaModelTests {
    @Test("Model loading")
    func modelLoading() throws {
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)

        #expect(model.pointer != nil, "Model should have valid pointer")
        #expect(model.embeddingDimensions > 0, "Model should have embedding dimensions")
        #expect(model.layers > 0, "Model should have layers")
        #expect(model.parameters > 0, "Model should have parameters")
        #expect(model.size > 0, "Model should have size")
    }

    @Test("Model loading with invalid path throws error")
    func modelInvalidPath() throws {
        SLlama.initialize()
        defer { SLlama.cleanup() }

        #expect(throws: SLlamaError.self) {
            _ = try SLlamaModel(modelPath: "/invalid/path/to/model.gguf")
        }
    }

    @Test("Model loading with null path throws error")
    func modelNullPath() throws {
        SLlama.initialize()
        defer { SLlama.cleanup() }

        #expect(throws: SLlamaError.self) {
            _ = try SLlamaModel(modelPath: "")
        }
    }

    @Test("Model properties access")
    func modelProperties() throws {
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)

        // Test basic properties
        #expect(model.embeddingDimensions > 0, "Model should have positive embedding dimensions")
        #expect(model.layers > 0, "Model should have positive layer count")
        #expect(model.attentionHeads > 0, "Model should have positive attention heads")
        #expect(model.parameters > 0, "Model should have positive parameter count")
        #expect(model.size > 0, "Model should have positive size")

        // Test computed properties
        #expect(model.vocab != nil, "Model should have vocabulary")
        #expect(model.trainingContextLength > 0, "Model should have training context length")
    }

    @Test("Model advanced properties")
    func modelAdvancedProperties() throws {
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)

        // Test advanced model features
        let advanced = model.advanced()
        #expect(advanced.validateModel(), "Model should pass validation")
        #expect(advanced.isCompatible(), "Model should be compatible")

        // Test metadata access
        if let metadata = advanced.getMetadata() {
            #expect(!metadata.isEmpty, "Model metadata should not be empty")
        }

        // Test dimensions
        if let dimensions = advanced.getDimensions() {
            #expect(dimensions["embedding_dimension"] == model.embeddingDimensions, "Embedding dimensions should match")
            #expect(dimensions["num_layers"] == model.layers, "Layer count should match")
        }
    }
}
