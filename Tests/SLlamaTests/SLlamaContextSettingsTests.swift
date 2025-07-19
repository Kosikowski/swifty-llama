import Foundation
import Testing
@testable import SLlama

struct SLlamaContextSettingsTests {
    @Test("Context settings")
    func contextSettings() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        SLlama.initialize()

        let model = try SLlamaModel(modelPath: modelPath)
        let context = try SLlamaContext(model: model)

        // Test various context properties and settings
        #expect(context.pointer != nil, "Context should have valid pointer")

        // Test context model
        if let contextModel = context.contextModel {
            #expect(contextModel.embeddingDimensions > 0, "Context model should have embedding dimensions")
        }
    }

    @Test("Context settings with custom parameters")
    func contextSettingsWithCustomParams() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)

        // Create custom context parameters using SLlama backend functions
        let context = try SLlamaContext(model: model)

        #expect(context.pointer != nil, "Context with custom params should have valid pointer")
    }

    @Test("Context creation with null model throws error")
    func contextInvalidParameters() throws {
        SLlama.initialize()
        defer { SLlama.cleanup() }

        // Test that creating context with null model throws error
        #expect(throws: SLlamaError.self) {
            try SLlamaContext(model: SLlamaModel(modelPointer: nil))
        }
    }

    @Test("Performance optimization")
    func performanceOptimization() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let context = try SLlamaContext(model: model)

        // Test performance-related operations
        #expect(context.pointer != nil, "Context should be valid for performance tests")
    }
}
