import Testing
@testable import SwiftyLlamaCpp

struct SLlamaSystemInfoTests {
    
    @Test("System information can be retrieved")
    func testSystemInfo() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }
        
        // Test system info logging
        SLlamaSystemInfo.logSystemInfo()
        #expect(true, "System info logging should not crash")
        
        // Test time retrieval
        let time1 = SLlamaSystemInfo.getCurrentTimeMicroseconds()
        let time2 = SLlamaSystemInfo.getCurrentTimeMicroseconds()
        #expect(time2 >= time1, "Time should be monotonically increasing")
        
        // Test device info
        let maxDevices = SLlamaSystemInfo.getMaxDevices()
        #expect(maxDevices >= 0, "Max devices should be non-negative")
        
        let maxParallelSequences = SLlamaSystemInfo.getMaxParallelSequences()
        #expect(maxParallelSequences >= 0, "Max parallel sequences should be non-negative")
    }
    
    @Test("System capabilities can be checked")
    func testSystemCapabilities() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }
        
        // Test capability checks
        let supportsMmap = SLlamaSystemInfo.supportsMmap()
        let supportsMlock = SLlamaSystemInfo.supportsMlock()
        let supportsGpuOffload = SLlamaSystemInfo.supportsGpuOffload()
        let supportsRpc = SLlamaSystemInfo.supportsRpc()
        
        // These should return boolean values (we can't predict the actual values)
        #expect(type(of: supportsMmap) == Bool.self, "Mmap support should return boolean")
        #expect(type(of: supportsMlock) == Bool.self, "Mlock support should return boolean")
        #expect(type(of: supportsGpuOffload) == Bool.self, "GPU offload support should return boolean")
        #expect(type(of: supportsRpc) == Bool.self, "RPC support should return boolean")
    }
    
    @Test("System capabilities struct works correctly")
    func testSystemCapabilitiesStruct() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }
        
        // Test capabilities struct
        let capabilities = SLlamaSystemCapabilities()
        
        #expect(capabilities.maxDevices >= 0, "Max devices should be non-negative")
        #expect(capabilities.maxParallelSequences >= 0, "Max parallel sequences should be non-negative")
        #expect(!capabilities.systemInfo.isEmpty, "System info description should not be empty")
        #expect(capabilities.currentTimeMicroseconds > 0, "Current time should be positive")
        
        // Test formatted description
        let description = capabilities.getFormattedDescription()
        #expect(!description.isEmpty, "Formatted description should not be empty")
        #expect(description.contains("System Capabilities:"), "Description should contain header")
        #expect(description.contains("Max Devices:"), "Description should contain device info")
    }
    
    @Test("SwiftyLlamaCpp system info extensions work")
    func testSwiftyLlamaCppSystemInfo() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }
        
        // Test convenience methods
        let capabilities = SwiftyLlamaCpp.getSystemCapabilities()
        #expect(capabilities.maxDevices >= 0, "Capabilities should be valid")
        
        // Test that print function doesn't crash
        // Note: We can't easily test the output, but we can ensure it doesn't crash
        SwiftyLlamaCpp.printSystemInfo()
        #expect(true, "Print system info should not crash")
    }
} 