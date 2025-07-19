import Testing
@testable import SLlama

struct SLlamaContextTests {
    @Test("Context enum types are accessible")
    func contextEnumTypes() throws {
        // Test that context-related enums are properly defined
        let poolingTypes: [SLlamaPoolingType] = [.none, .mean, .cls]
        for poolingType in poolingTypes {
            #expect(poolingType.rawValue >= 0, "Pooling type should have valid raw value")
        }

        // This is a minimal test that validates enum accessibility
        // Real context functionality requires a loaded model
        #expect(SLlamaPoolingType.none != .mean, "Enum values should be distinct")
    }
}
