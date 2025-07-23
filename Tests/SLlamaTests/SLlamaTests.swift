import Testing
@testable import SLlama
@testable import TestUtilities

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

    @Test("SLlama backend selection test")
    func backendSelectionTest() async throws {
        #expect(
            TestUtilities.isTestModelAvailable(),
            "Test model must be available for backend selection test"
        )

        let modelPath = TestUtilities.testModelPath

        // Test CPU backend
        let cpuModel = try SLlamaModel(modelPath: modelPath, backendType: .cpu)
        #expect(cpuModel.getBackendType() == .cpu, "CPU backend should be set")
        #expect(!cpuModel.isUsingGpuAcceleration(), "CPU backend should not use GPU acceleration")

        // Test GPU backend
        let gpuModel = try SLlamaModel(modelPath: modelPath, backendType: .gpu)
        #expect(gpuModel.getBackendType() == .gpu, "GPU backend should be set")
        // GPU acceleration depends on hardware support
        let gpuAccelerationAvailable = SLlama.supportsMetal() || SLlama.supportsGpuOffload()
        #expect(
            gpuModel.isUsingGpuAcceleration() == gpuAccelerationAvailable,
            "GPU acceleration should match hardware support"
        )

        // Test Auto backend
        let autoModel = try SLlamaModel(modelPath: modelPath, backendType: .auto)
        #expect(autoModel.getBackendType() == .auto, "Auto backend should be set")
        // Auto should use GPU if available, otherwise CPU
        let autoShouldUseGpu = SLlama.supportsMetal() || SLlama.supportsGpuOffload()
        #expect(autoModel.isUsingGpuAcceleration() == autoShouldUseGpu, "Auto backend should use GPU if available")

        // Test default initializer (should use auto)
        let defaultModel = try SLlamaModel(modelPath: modelPath)
        #expect(defaultModel.getBackendType() == .auto, "Default initializer should use auto backend")

        // Test backend info
        let cpuInfo = cpuModel.getBackendInfo()
        #expect(cpuInfo.contains("CPU"), "CPU backend info should mention CPU")

        let gpuInfo = gpuModel.getBackendInfo()
        #expect(gpuInfo.contains("GPU"), "GPU backend info should mention GPU")

        let autoInfo = autoModel.getBackendInfo()
        #expect(autoInfo.contains("Auto"), "Auto backend info should mention Auto")
    }
}
