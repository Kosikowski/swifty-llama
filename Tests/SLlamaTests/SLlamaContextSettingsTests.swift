import Foundation
import Testing
@testable import SLlama

struct SLlamaContextSettingsTests {
    @Test("Context settings with default parameters")
    func contextSettings() throws {
        let modelPath = SLlamaTestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let context = try SLlamaContext(model: model)

        // Test basic properties are accessible
        #expect(context.contextSize > 0)
        #expect(context.batchSize > 0)
    }

    @Test("Context settings with custom parameters")
    func contextSettingsWithCustomParams() throws {
        let modelPath = SLlamaTestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)

        // Create custom context parameters using the Swift API
        let params = SLlamaContext.createParams(
            contextSize: 1024,
            batchSize: 256,
            physicalBatchSize: 256,
            maxSequences: 2,
            threads: 4,
            batchThreads: 4
        )

        let context = try SLlamaContext(model: model, contextParams: params)

        // Verify custom settings are applied
        #expect(context.contextSize == 1024)
        #expect(context.batchSize == 256)
        #expect(context.maxBatchSize == 256)
        #expect(context.maxSequences == 2)
    }

    @Test("Context settings validation")
    func contextSettingsValidation() throws {
        let modelPath = SLlamaTestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)

        // Test invalid context size (zero)
        do {
            let params = SLlamaContext.createParams(contextSize: 0)
            _ = try SLlamaContext(model: model, contextParams: params)
            #expect(Bool(false), "Should have thrown an error for zero context size")
        } catch SLlamaError.invalidParameters {
            // Expected error
        }

        // Test invalid batch size (zero)
        do {
            let params = SLlamaContext.createParams(batchSize: 0)
            _ = try SLlamaContext(model: model, contextParams: params)
            #expect(Bool(false), "Should have thrown an error for zero batch size")
        } catch SLlamaError.invalidParameters {
            // Expected error
        }
    }
}
