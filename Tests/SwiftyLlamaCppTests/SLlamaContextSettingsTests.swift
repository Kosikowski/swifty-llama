import Testing
@testable import SwiftyLlamaCpp

struct SLlamaContextSettingsTests {
    
    @Test("Context settings work")
    func testContextSettings() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
        guard let model = SLlamaModel(modelPath: modelPath) else {
            // Failed to load test model
            return
        }
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            // Failed to create context
            return
        }
        
        // Test embeddings setting
        context.setEmbeddings(true)
        context.setEmbeddings(false)
        
        // Test causal attention setting
        context.setCausalAttention(true)
        context.setCausalAttention(false)
        
        // Test warmup setting
        context.setWarmup(true)
        context.setWarmup(false)
        
        // Test synchronization
        context.synchronize()
        
        // Test thread settings
        context.setThreads(nThreads: 4, nThreadsBatch: 2)
        
        // Test inference wrapper methods
        let inference = context.inference()
        inference.setEmbeddings(true)
        inference.setWarmup(true)
        inference.synchronize()
    }
    
    @Test("Performance optimization works")
    func testPerformanceOptimization() throws {
        let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
        guard let model = SLlamaModel(modelPath: modelPath) else {
            // Failed to load test model
            return
        }
        
        // Create context
        guard let context = SLlamaContext(model: model) else {
            // Failed to create context
            return
        }
        
        // Test performance optimization
        context.optimizeForPerformance(
            useOptimalThreads: true,
            customThreads: nil
        )
    }
} 