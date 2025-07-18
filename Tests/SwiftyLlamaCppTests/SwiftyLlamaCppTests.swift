import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct SwiftyLlamaCppTests {
    
    @Test("SwiftyLlamaCpp static functions")
    func testSwiftyLlamaCppStaticFunctions() throws {
        // Test initialization and cleanup
        SwiftyLlamaCpp.initialize()
        SwiftyLlamaCpp.free()
        
        // Test system info
        let systemInfo = SwiftyLlamaCpp.getSystemInfo()
        #expect(!systemInfo.isEmpty, "System info should not be empty")
        
        // Test capability checks
        let supportsMmap = SwiftyLlamaCpp.supportsMmap()
        let supportsMlock = SwiftyLlamaCpp.supportsMlock()
        let supportsGpuOffload = SwiftyLlamaCpp.supportsGpuOffload()
        
        #expect(type(of: supportsMmap) == Bool.self, "supportsMmap should return Bool")
        #expect(type(of: supportsMlock) == Bool.self, "supportsMlock should return Bool")
        #expect(type(of: supportsGpuOffload) == Bool.self, "supportsGpuOffload should return Bool")
        
        // Test device limits
        let maxDevices = SwiftyLlamaCpp.maxDevices()
        let maxParallelSequences = SwiftyLlamaCpp.maxParallelSequences()
        
        #expect(maxDevices >= 0, "maxDevices should be non-negative")
        #expect(maxParallelSequences >= 0, "maxParallelSequences should be non-negative")
    }
} 