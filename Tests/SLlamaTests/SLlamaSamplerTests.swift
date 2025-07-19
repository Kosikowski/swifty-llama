import Testing
@testable import SLlama

struct SLlamaSamplerTests {
    @Test("SLlamaSampler type exists and is accessible")
    func samplerTypeExists() throws {
        // Test that the SLlamaSampler type is properly defined
        #expect(SLlamaSampler.self == SLlamaSampler.self, "SLlamaSampler type should exist")
    }

    @Test("SLlamaSamplerChain type exists and is accessible")
    func samplerChainTypeExists() throws {
        // Test that the SLlamaSamplerChain type is properly defined
        #expect(SLlamaSamplerChain.self == SLlamaSamplerChain.self, "SLlamaSamplerChain type should exist")
    }
}
