import Testing
@testable import SwiftyLlamaCpp

struct SLlamaContextSettingsTests {
    
    @Test("Context settings functions can be called without crashing")
    func testContextSettings() throws {
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
        
        // Test embeddings setting
        context.setEmbeddings(true)
        context.setEmbeddings(false)
        #expect(true, "setEmbeddings should work without crashing")
        
        // Test causal attention setting
        context.setCausalAttention(true)
        context.setCausalAttention(false)
        #expect(true, "setCausalAttention should work without crashing")
        
        // Test warmup setting
        context.setWarmup(true)
        context.setWarmup(false)
        #expect(true, "setWarmup should work without crashing")
        
        // Test synchronization
        context.synchronize()
        #expect(true, "synchronize should work without crashing")
        
        // Test thread settings
        context.setThreads(nThreads: 4, nThreadsBatch: 2)
        #expect(true, "setThreads should work without crashing")
        
        // Test inference wrapper methods
        let inference = context.inference()
        inference.setEmbeddings(true)
        inference.setCausalAttention(true)
        inference.setWarmup(true)
        inference.synchronize()
        #expect(true, "Inference wrapper methods should work without crashing")
    }
    
    @Test("Context configuration for optimal performance works")
    func testOptimalPerformanceConfiguration() throws {
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
        
        // Test optimal performance configuration
        context.configureForOptimalPerformance(
            useOptimalThreads: true,
            enableEmbeddings: false,
            enableCausalAttention: true,
            enableWarmup: true
        )
        #expect(true, "configureForOptimalPerformance should work without crashing")
    }
} 