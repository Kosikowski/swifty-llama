import Foundation
import Testing
@testable import SwiftyLlamaCpp

class DummyContext: LlamaContext {
    override init?(model: LlamaModel, contextParams: LlamaContextParams? = nil) {
        super.init(model: model, contextParams: contextParams)
    }
    
    override var pointer: LlamaContextPointer? { nil }
    override var associatedModel: LlamaModel? { nil }
}

// MARK: - Test Utilities

class TestUtilities {
    
    /// Create a dummy context for testing purposes
    /// - Returns: A DummyContext instance, or nil if creation fails
    static func createDummyContext() -> DummyContext? {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
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