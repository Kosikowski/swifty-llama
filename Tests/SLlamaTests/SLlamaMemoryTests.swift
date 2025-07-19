import Foundation
import Testing
@testable import SLlama

struct SLlamaMemoryTests {
    @Test("SLlamaMemoryManager type exists and is accessible")
    func memoryManagerTypeExists() throws {
        // Test that the SLlamaMemoryManager type is properly defined
        // This is a minimal test since memory management requires a valid context
        #expect(SLlamaMemoryManager.self == SLlamaMemoryManager.self, "SLlamaMemoryManager type should exist")
    }
}
