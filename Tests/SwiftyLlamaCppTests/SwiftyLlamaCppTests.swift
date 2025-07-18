import Testing
import SwiftyLlamaCpp

@testable import SwiftyLlamaCpp

struct SwiftyLlamaCppTests {
    
    @Test("SwiftyLlamaCpp initialization and cleanup")
    func testSwiftyLlamaCppInitAndCleanup() throws {
        // Test initialization and cleanup
        SwiftyLlamaCpp.initialize()
        SwiftyLlamaCpp.cleanup()
        
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
        
        #expect(maxDevices >= 0, "maxDevices should be non-negative")
        #expect(maxParallelSequences >= 0, "maxParallelSequences should be non-negative")
    }
} 