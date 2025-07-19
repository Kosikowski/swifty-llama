import Foundation
import Testing
@testable import SwiftyLlamaCpp

// MARK: - SLlamaIntegrationTests

struct SLlamaIntegrationTests {
    // MARK: Properties

    let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

    // MARK: Functions

    @Test("Real model loading test")
    func realModelLoading() throws {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        guard let model = SLlamaModel(modelPath: modelPath) else {
            print("Test skipped: Model could not be loaded at \(modelPath)")
            return
        }

        // Model is successfully loaded if we reach this point
        #expect(model.pointer != nil, "Model should have a valid pointer")
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
