import Foundation
import Testing
@testable import SLlama

// MARK: - SLlamaIntegrationTests

struct SLlamaIntegrationTests {
    @Test("Basic model loading integration test")
    func basicModelLoadingIntegrationTest() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        // Check if test model exists
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        // Initialize the backend
        SLlama.initialize()

        let model = try SLlamaModel(modelPath: modelPath)

        // Test that model loads successfully (using actual property names)
        #expect(model.embeddingDimensions > 0, "Model should have embedding dimensions")
        #expect(model.trainingContextLength > 0, "Model should have training context length")

        // Test context creation
        let context = try SLlamaContext(model: model)
        #expect(context.pointer != nil, "Context should be valid")

        // Test model properties
        #expect(model.size > 0, "Model size should be positive")
        #expect(model.parameters > 0, "Model should have parameters")

        print("Integration test passed: Model loaded and context created successfully")

        // Cleanup
        SLlama.cleanup()
    }

    @Test("Model with invalid path throws appropriate error")
    func modelInvalidPathIntegrationTest() throws {
        SLlama.initialize()

        // Test that invalid path throws expected error
        #expect(throws: SLlamaError.self) {
            try SLlamaModel(modelPath: "/definitely/not/a/real/path.gguf")
        }

        SLlama.cleanup()
    }
}

// MARK: - TestError

enum TestError: Error {
    case modelNotFound
    case modelLoadFailed
    case contextCreationFailed
    case vocabLoadFailed
    case tokenizationFailed
    case detokenizationFailed
}
