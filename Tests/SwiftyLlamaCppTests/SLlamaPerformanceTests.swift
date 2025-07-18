import Testing
@testable import SwiftyLlamaCpp

struct SLlamaPerformanceTests {
    
    @Test("Performance functions can be called without crashing")
    func testPerformanceFunctions() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }
        
        // Load test model
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
        guard let model = SLlamaModel(modelPath: modelPath) else {
            #expect(false, "Failed to load test model")
            return
        }
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            #expect(false, "Failed to create context")
            return
        }
        
        // Test that performance functions can be called without crashing
        let performance = SLlamaPerformance()
        
        // Test context performance functions
        let contextData = performance.getContextPerformanceData(context: context)
        // Note: This will be nil since llama.cpp performance functions aren't available
        #expect(contextData == nil, "Context performance data should be nil when llama.cpp functions aren't available")
        
        // Test that print functions don't crash
        performance.printContextPerformance(context: context)
        performance.resetContextPerformance(context: context)
        
        // Test sampler performance functions
        let sampler = SLlamaSampler(context: context)
        let samplerData = performance.getSamplerPerformanceData(sampler: sampler)
        #expect(samplerData == nil, "Sampler performance data should be nil when llama.cpp functions aren't available")
        
        // Test that sampler print functions don't crash
        performance.printSamplerPerformance(sampler: sampler)
        performance.resetSamplerPerformance(sampler: sampler)
    }
    
    @Test("Detailed performance metrics return valid data")
    func testDetailedPerformanceMetrics() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }
        
        // Load test model
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
        guard let model = SLlamaModel(modelPath: modelPath) else {
            #expect(false, "Failed to load test model")
            return
        }
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            #expect(false, "Failed to create context")
            return
        }
        
        let performance = SLlamaPerformance()
        
        // Test context metrics
        let contextMetrics = performance.getDetailedContextMetrics(context: context)
        #expect(contextMetrics.startTimeMs >= 0, "Start time should be non-negative")
        #expect(contextMetrics.loadTimeMs >= 0, "Load time should be non-negative")
        #expect(contextMetrics.promptEvalTimeMs >= 0, "Prompt eval time should be non-negative")
        #expect(contextMetrics.evalTimeMs >= 0, "Eval time should be non-negative")
        #expect(contextMetrics.promptEvalCount >= 0, "Prompt eval count should be non-negative")
        #expect(contextMetrics.evalCount >= 0, "Eval count should be non-negative")
        #expect(contextMetrics.reusedCount >= 0, "Reused count should be non-negative")
        
        // Test sampler metrics
        let sampler = SLlamaSampler(context: context)
        let samplerMetrics = performance.getDetailedSamplerMetrics(sampler: sampler)
        #expect(samplerMetrics.sampleTimeMs >= 0, "Sample time should be non-negative")
        #expect(samplerMetrics.sampleCount >= 0, "Sample count should be non-negative")
        #expect(samplerMetrics.averageSampleTimeMs >= 0, "Average sample time should be non-negative")
        #expect(samplerMetrics.samplesPerSecond >= 0, "Samples per second should be non-negative")
    }
    
    @Test("Context performance extension methods work")
    func testContextPerformanceExtensions() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }
        
        // Load test model
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
        guard let model = SLlamaModel(modelPath: modelPath) else {
            #expect(false, "Failed to load test model")
            return
        }
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            #expect(false, "Failed to create context")
            return
        }
        
        // Test performance interface
        let performance = context.performance()
        #expect(performance is SLlamaPerformance, "Should return SLlamaPerformance instance")
        
        // Test detailed metrics
        let metrics = context.getDetailedPerformanceMetrics()
        #expect(metrics.startTimeMs >= 0, "Start time should be non-negative")
        #expect(metrics.totalEvalTimeMs >= 0, "Total eval time should be non-negative")
    }
    
    @Test("Sampler performance extension methods work")
    func testSamplerPerformanceExtensions() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }
        
        // Load test model
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
        guard let model = SLlamaModel(modelPath: modelPath) else {
            #expect(false, "Failed to load test model")
            return
        }
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            #expect(false, "Failed to create context")
            return
        }
        
        let sampler = SLlamaSampler(context: context)
        
        // Test detailed metrics
        let metrics = sampler.getDetailedPerformanceMetrics()
        #expect(metrics.sampleTimeMs >= 0, "Sample time should be non-negative")
        #expect(metrics.sampleCount >= 0, "Sample count should be non-negative")
        #expect(metrics.averageSampleTimeMs >= 0, "Average sample time should be non-negative")
    }
    
    @Test("Performance monitoring can be started")
    @MainActor
    func testPerformanceMonitoring() throws {
        // Initialize backend
        SLlamaBackend.initialize()
        defer { SLlamaBackend.free() }
        
        // Load test model
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
        guard let model = SLlamaModel(modelPath: modelPath) else {
            #expect(false, "Failed to load test model")
            return
        }
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            #expect(false, "Failed to create context")
            return
        }
        
        // Test that monitoring can be started
        let monitor = context.startPerformanceMonitoring()
        #expect(monitor != nil, "Performance monitoring should start successfully")
        
        // Test that monitor provides metrics
        if let monitor = monitor {
            let results = monitor.getResults()
            #expect(results.count >= 0, "Results array should be accessible")
            
            // Test that we can start and stop monitoring
            monitor.start()
            monitor.stop()
            #expect(true, "Monitor start/stop should work without crashing")
        }
    }
} 