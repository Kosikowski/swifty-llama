import Testing
@testable import SLlama

struct SLlamaQuantizationTests {
    @Test("Default quantization parameters are valid")
    func defaultQuantizationParams() throws {
        let defaultParams = SLlamaQuantization.defaultParams()

        // Validate default parameter values
        #expect(defaultParams.nthread >= 0, "Default thread count should be non-negative")
        #expect(defaultParams.nthread <= 64, "Default thread count should be reasonable")
        #expect(defaultParams.ftype.rawValue >= 0, "Default file type should be valid")
        #expect(defaultParams.allow_requantize == false, "Default should not allow requantize")
        #expect(defaultParams.quantize_output_tensor == true, "Default should quantize output tensor")
        #expect(defaultParams.only_copy == false, "Default should not be copy-only")
        #expect(defaultParams.pure == false, "Default should not be pure")
        #expect(defaultParams.keep_split == false, "Default should not keep split")
    }

    @Test("Custom quantization parameters work correctly")
    func customQuantizationParams() throws {
        // Test basic quantization file types that should exist
        let fileTypes: [SLlamaFileType] = [
            .mostlyF16, .mostlyQ4_0, .mostlyQ4_1, .mostlyQ5_0,
            .mostlyQ5_1, .mostlyQ8_0, .allF32,
        ]

        for fileType in fileTypes {
            let customParams = SLlamaQuantization.createParams(
                fileType: fileType,
                threads: 8,
                allowRequantize: true,
                quantizeOutputTensor: false,
                onlyCopy: true,
                pure: true,
                keepSplit: true
            )

            // Validate all parameters are set correctly
            #expect(customParams.ftype == fileType, "File type should be set correctly for \(fileType)")
            #expect(customParams.nthread == 8, "Thread count should be set correctly")
            #expect(customParams.allow_requantize == true, "Allow requantize should be set correctly")
            #expect(customParams.quantize_output_tensor == false, "Quantize output tensor should be set correctly")
            #expect(customParams.only_copy == true, "Only copy should be set correctly")
            #expect(customParams.pure == true, "Pure should be set correctly")
            #expect(customParams.keep_split == true, "Keep split should be set correctly")
        }
    }

    @Test("Quantization parameter edge cases")
    func quantizationParameterEdgeCases() throws {
        // Test with zero threads (should use default)
        let zeroThreadParams = SLlamaQuantization.createParams(
            fileType: .mostlyQ4_0,
            threads: 0
        )
        #expect(zeroThreadParams.nthread >= 0, "Zero threads should result in valid thread count")

        // Test with negative threads (passed through as-is, not validated)
        let negativeThreadParams = SLlamaQuantization.createParams(
            fileType: .mostlyQ4_0,
            threads: -1
        )
        #expect(negativeThreadParams.nthread == -1, "Negative threads are passed through as-is")

        // Test with very high thread count
        let highThreadParams = SLlamaQuantization.createParams(
            fileType: .mostlyQ4_0,
            threads: 1000
        )
        #expect(highThreadParams.nthread == 1000, "High thread count should be passed through")
    }

    @Test("Quantization file type validation")
    func quantizationFileTypeValidation() throws {
        // Test common file types that should exist
        let commonFileTypes: [SLlamaFileType] = [
            .allF32, .mostlyF16, .mostlyQ4_0, .mostlyQ4_1,
            .mostlyQ8_0, .mostlyQ5_0, .mostlyQ5_1, .mostlyQ2_K, .mostlyQ6_K,
        ]

        for fileType in commonFileTypes {
            let params = SLlamaQuantization.createParams(fileType: fileType)
            #expect(params.ftype == fileType, "File type \(fileType) should be set correctly")
            #expect(params.ftype.rawValue >= 0, "File type \(fileType) should have valid raw value")
        }
    }

    @Test("Backend initialization for quantization")
    @SLlamaActor
    func backendInitializationForQuantization() async throws {
        // Test that backend can be initialized for quantization operations
        SLlamaBackend.initialize()
        defer {
            SLlamaBackend.free()
        }

        // Verify backend is initialized
        let isInitialized = SLlamaBackend.isInitialized
        #expect(isInitialized == true, "Backend should be initialized")

        // Test that we can create parameters after initialization
        let params = SLlamaQuantization.defaultParams()
        #expect(params.nthread >= 0, "Parameters should be valid after backend initialization")
    }

    @Test("Quantization method overloads work correctly")
    func quantizationMethodOverloads() throws {
        // Test createParams with minimal parameters
        let minimalParams = SLlamaQuantization.createParams(fileType: .mostlyQ4_0)
        #expect(minimalParams.ftype == .mostlyQ4_0, "Minimal params should set file type")

        // Test createParams with partial parameters
        let partialParams = SLlamaQuantization.createParams(
            fileType: .mostlyQ8_0,
            threads: 4
        )
        #expect(partialParams.ftype == .mostlyQ8_0, "Partial params should set file type")
        #expect(partialParams.nthread == 4, "Partial params should set thread count")

        // Test createParams with all boolean parameters
        let boolParams = SLlamaQuantization.createParams(
            fileType: .mostlyQ5_0,
            allowRequantize: true,
            quantizeOutputTensor: false,
            onlyCopy: true,
            pure: true,
            keepSplit: true
        )
        #expect(boolParams.ftype == .mostlyQ5_0, "Boolean params should set file type")
        #expect(boolParams.allow_requantize == true, "Boolean params should set allow requantize")
        #expect(boolParams.quantize_output_tensor == false, "Boolean params should set quantize output tensor")
        #expect(boolParams.only_copy == true, "Boolean params should set only copy")
        #expect(boolParams.pure == true, "Boolean params should set pure")
        #expect(boolParams.keep_split == true, "Boolean params should set keep split")
    }

    @Test("Quantization parameter combinations")
    func quantizationParameterCombinations() throws {
        // Test different parameter combinations that should work
        let combinations: [(SLlamaFileType, Int32, Bool, Bool)] = [
            (.allF32, 1, false, false),
            (.mostlyF16, 2, true, false),
            (.mostlyQ4_0, 4, false, true),
            (.mostlyQ4_1, 8, true, true),
            (.mostlyQ5_0, 16, false, false),
            (.mostlyQ8_0, 0, true, false), // 0 threads should work
        ]

        for (fileType, threads, allowRequantize, quantizeOutput) in combinations {
            let params = SLlamaQuantization.createParams(
                fileType: fileType,
                threads: threads,
                allowRequantize: allowRequantize,
                quantizeOutputTensor: quantizeOutput
            )

            #expect(params.ftype == fileType, "File type should match for combination")
            #expect(params.nthread >= 0, "Thread count should be non-negative for combination")
            #expect(params.allow_requantize == allowRequantize, "Allow requantize should match")
            #expect(params.quantize_output_tensor == quantizeOutput, "Quantize output should match")
        }
    }
}
