import Testing
@testable import SLlama

struct SLlamaQuantizationTests {
    @Test("Quantization parameters can be created")
    func quantizationParams() throws {
        // Test default parameters
        let defaultParams = SLlamaQuantization.defaultParams()
        #expect(defaultParams.nthread >= 0, "Default thread count should be non-negative")

        // Test custom parameters
        let customParams = SLlamaQuantization.createParams(
            fileType: .mostlyQ4_0,
            threads: 4,
            allowRequantize: true,
            quantizeOutputTensor: true,
            onlyCopy: false,
            pure: false,
            keepSplit: false
        )
        #expect(customParams.ftype == .mostlyQ4_0, "File type should be set correctly")
        #expect(customParams.nthread == 4, "Thread count should be set correctly")
        #expect(customParams.allow_requantize == true, "Allow requantize should be set correctly")
        #expect(customParams.quantize_output_tensor == true, "Quantize output tensor should be set correctly")
    }

    @Test("Quantization functions can be called without crashing")
    @MainActor
    func quantizationFunctions() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }

        // Test that quantization functions exist and can be called
        // Note: We can't actually test quantization without a real model file
        // This test just ensures the functions are available and parameters are valid
        // In a real-world scenario, you would need a test model file to verify quantization

        let defaultParams = SLlamaQuantization.defaultParams()
        #expect(defaultParams.nthread >= 0, "Default parameters should be valid")

        // Test that we can create custom parameters
        let customParams = SLlamaQuantization.createParams(
            fileType: .mostlyQ4_1,
            threads: 2,
            allowRequantize: false,
            quantizeOutputTensor: true
        )
        #expect(customParams.ftype == .mostlyQ4_1, "Custom parameters should be set correctly")
    }
}
