import Testing
import SwiftyLlamaCpp

@testable import SwiftyLlamaCpp

struct SwiftyLlamaCppTests {
    
    @Test("SwiftyLlamaCpp initialization and cleanup")
    func testSwiftyLlamaCppInitAndCleanup() throws {
        // Disable logging to suppress verbose output
        SwiftyLlamaCpp.disableLogging()
        
        // Test initialization and cleanup
        SwiftyLlamaCpp.initialize()
        
        // Test system info
        let currentTime = SwiftyLlamaCpp.getCurrentTime()
        #expect(currentTime >= 0, "Current time should be non-negative")
        
        // Test support flags
        let supportsMmap = SwiftyLlamaCpp.supportsMmap()
        let supportsMlock = SwiftyLlamaCpp.supportsMlock()
        let supportsGPUOffload = SwiftyLlamaCpp.supportsGPUOffload()
        
        #expect(type(of: supportsMmap) == Bool.self, "supportsMmap should return Bool")
        #expect(type(of: supportsMlock) == Bool.self, "supportsMlock should return Bool")
        #expect(type(of: supportsGPUOffload) == Bool.self, "supportsGPUOffload should return Bool")
        
        // Test device limits
        let maxDevices = SwiftyLlamaCpp.getMaxDevices()
        let maxParallelSequences = SwiftyLlamaCpp.getMaxParallelSequences()
        
        #expect(maxDevices >= 0, "Max devices should be non-negative")
        #expect(maxParallelSequences >= 0, "Max parallel sequences should be non-negative")
        
        // Cleanup
        SwiftyLlamaCpp.cleanup()
    }
} 