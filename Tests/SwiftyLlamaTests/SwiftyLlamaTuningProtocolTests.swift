import Foundation
import Testing
@testable import SwiftyLlama
@testable import TestUtilities

@Suite("SwiftyTuningLlamaProtocol Tests")
@SwiftyLlamaActor
struct SwiftyLlamaTuningProtocolTests {
    @Test("SwiftyTuningLlamaProtocol conformance test")
    func protocolConformance() async throws {
        // Verify that SwiftyTuningLlama conforms to SwiftyTuningLlamaProtocol
        let tuningLlama: SwiftyLlamaTuning = SwiftyLlamaTuner()

        // Test that we can call protocol methods
        #expect(tuningLlama.getCurrentLoRA() == nil)
        #expect(tuningLlama.getAvailableAdapters().isEmpty)

        // Training methods now throw trainingNotSupported
        do {
            _ = try tuningLlama.getCurrentTrainingSession()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.getTrainingMetrics()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }
    }

    @Test("SwiftyTuningLlamaProtocol default parameter extensions test")
    func defaultParameterExtensions() async throws {
        let tuningLlama: SwiftyLlamaTuning = SwiftyLlamaTuner()

        // Test that default parameter extensions work
        // These should compile and not throw errors for the default implementations
        do {
            _ = try tuningLlama.prepareTrainingData(conversations: [], validationSplit: 0.1)
        } catch {
            // Expected to throw TuningError.tokenizerNotInitialized
            #expect(error is TuningError)
        }

        do {
            try tuningLlama.applyLoRA(path: "/nonexistent/adapter.gguf")
        } catch {
            // Expected to throw TuningError.contextNotInitialized
            #expect(error is TuningError)
        }
    }

    @Test("SwiftyTuningLlamaProtocol type safety test")
    func typeSafety() async throws {
        // Test that the protocol provides type safety
        let tuningLlama: SwiftyLlamaTuning = SwiftyLlamaTuner()

        // Verify that protocol methods return the correct types
        let currentLoRA = tuningLlama.getCurrentLoRA()
        #expect(currentLoRA == nil)

        let availableAdapters = tuningLlama.getAvailableAdapters()
        #expect(availableAdapters.isEmpty)

        // Training methods now throw trainingNotSupported
        do {
            _ = try tuningLlama.getCurrentTrainingSession()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.getTrainingMetrics()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }
    }

    @Test("Training not supported error")
    func trainingNotSupported() async throws {
        let tuningLlama: SwiftyLlamaTuning = SwiftyLlamaTuner()

        // All training-related methods should throw trainingNotSupported
        do {
            _ = try tuningLlama.startTrainingSession(
                dataset: TrainingDataset(training: [], validation: []),
                config: TrainingConfig()
            )
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            try tuningLlama.stopTrainingSession()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            try tuningLlama.saveLoRAAdapter(path: "/tmp/test.gguf")
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            try tuningLlama.saveFineTunedModel(path: "/tmp/test.gguf")
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            try tuningLlama.exportModel(path: "/tmp/test.gguf", format: .gguf)
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.getModelSavingInfo()
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }

        do {
            _ = try tuningLlama.evaluateModel(validationExamples: [])
            Issue.record("Expected trainingNotSupported error")
        } catch let error as TuningError {
            #expect(error == .trainingNotSupported)
        }
    }
}
