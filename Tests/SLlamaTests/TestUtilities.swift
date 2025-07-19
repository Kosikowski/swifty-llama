import Foundation
import Testing
@testable import SLlama

// MARK: - DummyContext

class DummyContext: SLlamaContext {
    // MARK: Overridden Properties

    override var pointer: SLlamaContextPointer? { nil }
    override var associatedModel: SLlamaModel? { nil }

    // MARK: Lifecycle

    override init?(model: SLlamaModel, contextParams: SLlamaContextParams? = nil) {
        super.init(model: model, contextParams: contextParams)
    }
}

// MARK: - TestUtilities

class TestUtilities {
    /// Create a dummy context for testing purposes
    /// - Returns: A DummyContext instance, or nil if creation fails
    static func createDummyContext() -> DummyContext? {
        let dummyModel = SLlamaModel(modelPath: "/nonexistent/path/model.gguf")
        guard let model = dummyModel else { return nil }
        return DummyContext(model: model)
    }

    /// Execute a test with a dummy context, handling creation failures gracefully
    /// - Parameter test: The test closure to execute with the dummy context
    /// - Returns: true if the test executed successfully, false if context creation failed
    static func withDummyContext(_ test: (DummyContext) -> Void) -> Bool {
        guard let ctx = createDummyContext() else { return false }
        test(ctx)
        return true
    }
}
