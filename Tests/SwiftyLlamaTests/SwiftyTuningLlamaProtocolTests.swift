import Foundation
import Testing
@testable import SwiftyLlama
@testable import TestUtilities

@Suite("SwiftyTuningLlamaProtocol Tests")
@SwiftyLlamaActor
struct SwiftyTuningLlamaProtocolTests {
    @Test("SwiftyTuningLlamaProtocol conformance test")
    func protocolConformance() async throws {
        // Verify that SwiftyTuningLlama conforms to SwiftyTuningLlamaProtocol
        let tuningLlama: SwiftyTuningLlamaProtocol = SwiftyTuningLlama()

        // Test that we can call protocol methods
        #expect(tuningLlama.getCurrentLoRA() == nil)
        #expect(tuningLlama.getAvailableAdapters().isEmpty)
        #expect(tuningLlama.getCurrentTrainingSession() == nil)
        #expect(tuningLlama.getTrainingMetrics() == nil)
    }

    @Test("SwiftyTuningLlamaProtocol default parameter extensions test")
    func defaultParameterExtensions() async throws {
        let tuningLlama: SwiftyTuningLlamaProtocol = SwiftyTuningLlama()

        // Test that default parameter extensions work
        // These should compile and not throw errors for the default implementations
        do {
            _ = try tuningLlama.prepareTrainingData(conversations: [])
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
        let tuningLlama: SwiftyTuningLlamaProtocol = SwiftyTuningLlama()

        // Verify that protocol methods return the correct types
        let currentLoRA = tuningLlama.getCurrentLoRA()
        #expect(currentLoRA == nil)

        let availableAdapters = tuningLlama.getAvailableAdapters()
        #expect(availableAdapters.isEmpty)

        let currentSession = tuningLlama.getCurrentTrainingSession()
        #expect(currentSession == nil)

        let trainingMetrics = tuningLlama.getTrainingMetrics()
        #expect(trainingMetrics == nil)
    }
}
