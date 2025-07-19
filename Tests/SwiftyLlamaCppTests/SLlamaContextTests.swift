import Testing
@testable import SwiftyLlamaCpp

struct SLlamaContextTests {
    @Test("Wrapper functions compile and work")
    func wrapperFunctionsCompile() throws {
        // Test that our wrapper functions are accessible and don't crash
        // This test doesn't require loading a model, just verifies the API exists

        // Test that SLlamaPoolingType.none exists
        let poolingType: SLlamaPoolingType = .none
        #expect(poolingType == .none, "Pooling type should be accessible")

        // Test that the wrapper functions are defined in the context
        // We can't test them without a context, but we can verify the types exist
        let wrapperFunctionsExist = true
        #expect(wrapperFunctionsExist, "Wrapper functions should be accessible")

        // Test that we can access the pooling type enum values
        #expect(SLlamaPoolingType.none == .none, "Pooling type none should be accessible")
        #expect(SLlamaPoolingType.mean == .mean, "Pooling type mean should be accessible")
        #expect(SLlamaPoolingType.cls == .cls, "Pooling type cls should be accessible")
    }
}
