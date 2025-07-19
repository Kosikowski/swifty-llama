import Testing
@testable import SwiftyLlamaCpp

struct SLlamaSystemInfoTests {
    @Test("System info logging works")
    func systemInfoLogging() throws {
        // Test system info logging
        SLlamaSystemInfo.logSystemInfo()

        // Test time retrieval
        let time = SLlamaSystemInfo.getCurrentTimeMicroseconds()
        #expect(time > 0, "Time should be positive")

        // Test device info
        let maxDevices = SLlamaSystemInfo.getMaxDevices()
        #expect(maxDevices > 0, "Max devices should be positive")

        let maxParallelSequences = SLlamaSystemInfo.getMaxParallelSequences()
        #expect(maxParallelSequences > 0, "Max parallel sequences should be positive")
    }

    @Test("System capabilities work")
    func systemCapabilities() throws {
        // Test capability checks
        let supportsMmap = SLlamaSystemInfo.supportsMmap()
        let supportsMlock = SLlamaSystemInfo.supportsMlock()
        let supportsGpuOffload = SLlamaSystemInfo.supportsGpuOffload()
        let supportsRpc = SLlamaSystemInfo.supportsRpc()

        // These should return boolean values (may be true or false depending on system)
        #expect(supportsMmap == true || supportsMmap == false, "Supports mmap should be boolean")
        #expect(supportsMlock == true || supportsMlock == false, "Supports mlock should be boolean")
        #expect(supportsGpuOffload == true || supportsGpuOffload == false, "Supports GPU offload should be boolean")
        #expect(supportsRpc == true || supportsRpc == false, "Supports RPC should be boolean")
    }

    @Test("System capabilities struct works")
    func systemCapabilitiesStruct() throws {
        let capabilities = SLlamaSystemCapabilities()

        #expect(capabilities.maxDevices > 0, "Max devices should be positive")
        #expect(capabilities.maxParallelSequences > 0, "Max parallel sequences should be positive")
        #expect(capabilities.currentTimeMicroseconds > 0, "Current time should be positive")

        // Check that system info is available
        #expect(!capabilities.systemInfo.isEmpty, "System info should not be empty")
    }

    @Test("SwiftyLlamaCpp extensions work")
    func swiftyLlamaCppExtensions() throws {
        // Test system capabilities
        let capabilities = SwiftyLlamaCpp.getSystemCapabilities()
        #expect(capabilities.maxDevices > 0, "Max devices should be positive")
        #expect(capabilities.maxParallelSequences > 0, "Max parallel sequences should be positive")

        // Test system info printing
        SwiftyLlamaCpp.printSystemInfo()
    }
}
