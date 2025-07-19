import Foundation
import Testing
@testable import SLlama

struct SLlamaModelTests {
    @Test("Model initialization")
    func sLlamaModelInitialization() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        // Check if test model exists
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()

        let model = try SLlamaModel(modelPath: modelPath)

        // Basic model properties (using actual property names)
        #expect(model.embeddingDimensions > 0, "Model should have embedding dimensions")
        #expect(model.layers > 0, "Model should have layers")
        #expect(model.parameters > 0, "Model should have parameters")
        #expect(model.size > 0, "Model should have size")
        #expect(model.trainingContextLength > 0, "Model should have training context length")

        // Test token properties via vocab
        if let vocab = model.vocab {
            let vocabWrapper = SLlamaVocab(vocab: vocab)
            let bosToken = vocabWrapper.bosToken
            let eosToken = vocabWrapper.eosToken
            #expect(bosToken >= 0, "BOS token should be non-negative")
            #expect(eosToken >= 0, "EOS token should be non-negative")
            #expect(vocabWrapper.tokenCount > 0, "Vocab should have tokens")
        }

        // Test description with new throwing API - let it throw if it fails
        let description = try model.description()
        #expect(!description.isEmpty, "Model description should not be empty")
    }

    @Test("Model metadata access")
    func sLlamaModelMetadata() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        // Check if test model exists
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)

        // Test metadata count
        let metadataCount = model.metadataCount
        #expect(metadataCount >= 0, "Metadata count should be non-negative")

        // Test accessing metadata by index with new throwing API
        // Only test if metadata exists
        if metadataCount > 0 {
            let key = try model.metadataKey(at: 0)
            #expect(!key.isEmpty, "First metadata key should not be empty")

            let value = try model.metadataValue(at: 0)
            #expect(!value.isEmpty, "First metadata value should not be empty")
        }

        // Test accessing specific metadata by key - this may legitimately fail
        // so we use try? for optional testing
        if let architecture = try? model.metadataValue(for: "general.architecture") {
            #expect(!architecture.isEmpty, "Architecture metadata should not be empty")
        }
    }

    @Test("Model initialization with invalid path throws error")
    func sLlamaModelInvalidPath() throws {
        SLlama.initialize()
        defer { SLlama.cleanup() }

        // Test that invalid path throws an error
        #expect(throws: SLlamaError.self) {
            try SLlamaModel(modelPath: "/nonexistent/path/to/model.gguf")
        }
    }

    @Test("Model metadata access with invalid index throws error")
    func sLlamaModelInvalidMetadataAccess() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)

        // Test invalid index should throw
        #expect(throws: SLlamaError.self) {
            try model.metadataKey(at: -1)
        }

        #expect(throws: SLlamaError.self) {
            try model.metadataKey(at: 9999)
        }
    }
}
