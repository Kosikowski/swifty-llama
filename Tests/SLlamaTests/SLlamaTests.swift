import Testing
@testable import SLlama

struct SLlamaTests {
    @Test("SLlama initialization and cleanup")
    func SLlamaInitAndCleanup() throws {
        // Disable logging to suppress verbose output
        SLlama.disableLogging()

        // Test initialization and cleanup
        SLlama.initialize()

        // Test system info
        let currentTime = SLlama.getCurrentTime()
        #expect(currentTime >= 0, "Current time should be non-negative")

        // Test support flags
        let supportsMmap = SLlama.supportsMmap()
        let supportsMlock = SLlama.supportsMlock()
        let supportsGPUOffload = SLlama.supportsGPUOffload()

        #expect(type(of: supportsMmap) == Bool.self, "supportsMmap should return Bool")
        #expect(type(of: supportsMlock) == Bool.self, "supportsMlock should return Bool")
        #expect(type(of: supportsGPUOffload) == Bool.self, "supportsGPUOffload should return Bool")

        // Test device limits
        let maxDevices = SLlama.getMaxDevices()
        let maxParallelSequences = SLlama.getMaxParallelSequences()

        #expect(maxDevices >= 0, "Max devices should be non-negative")
        #expect(maxParallelSequences >= 0, "Max parallel sequences should be non-negative")

        // Cleanup
        SLlama.cleanup()
    }
}
