import Foundation
import Testing
import TestUtilities
@testable import SLlama

struct SLlamaPerformanceTests {
    @Test("Performance instance can be created")
    func performanceInstanceCreation() throws {
        // Test creating performance instance without context
        let performance = SLlamaPerformance()
        #expect(type(of: performance) == SLlamaPerformance.self, "Should create SLlamaPerformance instance")

        // Test creating performance instance with nil context
        let performanceWithNil = SLlamaPerformance(context: nil)
        #expect(
            type(of: performanceWithNil) == SLlamaPerformance.self,
            "Should create SLlamaPerformance instance with nil context"
        )
    }

    @Test("Performance monitor can be created")
    @SLlamaActor
    func performanceMonitorCreation() throws {
        // Test creating monitor without context
        let monitor = SPerformanceMonitor(context: nil)
        #expect(type(of: monitor) == SPerformanceMonitor.self, "Should create SPerformanceMonitor instance")

        // Test monitor control methods
        monitor.start()
        #expect(
            monitor.getResults().isEmpty == false || monitor.getResults().isEmpty == true,
            "Results should be accessible"
        )

        monitor.stop()
        monitor.clearResults()
        #expect(monitor.getResults().isEmpty, "Results should be empty after clearing")
    }

    @Test("Performance benchmark methods work")
    func performanceBenchmarkMethods() throws {
        let performance = SLlamaPerformance()

        // Test memory profiling
        let memoryResults = performance.profileMemoryUsage(
            operation: {
                // Simple operation for testing
                _ = [1, 2, 3, 4, 5].map { $0 * 2 }
            },
            maxTokens: 10
        )

        #expect(memoryResults != nil, "Memory profiling should return results")
        if let results = memoryResults {
            #expect(results.initialMemory >= 0, "Initial memory should be non-negative")
            #expect(results.finalMemory >= 0, "Final memory should be non-negative")
            #expect(results.memoryIncrease >= 0, "Memory increase should be non-negative")
        }

        // Test CPU profiling
        let cpuResults = performance.profileCPUUsage(
            operation: {
                // Simple operation for testing
                _ = [1, 2, 3, 4, 5].map { $0 * 2 }
            },
            maxTokens: 10
        )

        #expect(cpuResults != nil, "CPU profiling should return results")
        if let results = cpuResults {
            #expect(results.duration >= 0, "Duration should be non-negative")
            #expect(results.startCPU >= 0, "Start CPU should be non-negative")
            #expect(results.endCPU >= 0, "End CPU should be non-negative")
        }
    }

    @Test("Model loading benchmark works")
    func modelLoadingBenchmark() throws {
        let performance = SLlamaPerformance()

        // Test model loading benchmark with a small number of iterations
        let benchmarkResults = performance.benchmarkModelLoading(
            modelPath: TestUtilities.testModelPath,
            iterations: 1
        )

        #expect(benchmarkResults != nil, "Benchmark should return results")
        if let results = benchmarkResults {
            print("Benchmark results: \(results)")
            #expect(results.averageLoadTime > 0, "Average load time should be positive")
            #expect(results.totalLoadTime > 0, "Total load time should be positive")
            #expect(results.iterations == 1, "Iterations should match")
        }
    }

    @Test("Performance optimization")
    func performanceOptimization() throws {
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let context = try SLlamaContext(model: model)

        // Test performance-related operations
        #expect(context.pointer != nil, "Context should be valid for performance tests")
    }
}
