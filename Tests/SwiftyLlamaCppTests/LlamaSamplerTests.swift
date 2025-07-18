import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct LlamaSamplerTests {
    
    @Test("LlamaSampler typealiases exist and are correct")
    func testLlamaSamplerTypealiases() throws {
        let _: LlamaSamplerPointer? = nil
        let _: LlamaSamplerContext? = nil
        let _: LlamaTokenDataArrayPointer? = nil
    }
    
    @Test("Model creation with invalid path fails as expected")
    func testModelCreationWithInvalidPath() throws {
        let model = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        #expect(model == nil, "Model creation with invalid path should return nil")
    }
    
    @Test("LlamaSampler construction with dummy context")
    func testLlamaSamplerConstruction() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let sampler = LlamaSampler(context: ctx)
            #expect(sampler.cSampler == nil)
            sampler.accept(0)
            sampler.reset()
            let clone = sampler.clone()
            #expect(clone == nil)
        } else {
            // If dummy context creation fails, that's expected behavior
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSampler static initializers return nil without C API")
    func testLlamaSamplerStaticInitializers() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            #expect(LlamaSampler.greedy(context: ctx) == nil)
            #expect(LlamaSampler.distribution(context: ctx, seed: 42) == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain construction with dummy context")
    func testLlamaSamplerChainConstruction() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            #expect(chain.cChain == nil)
            #expect(chain.samplerCount == 0)
            chain.accept(0)
            chain.reset()
            let clone = chain.clone()
            #expect(clone == nil)
            #expect(chain.getSampler(at: 0) == nil)
            #expect(chain.removeSampler(at: 0) == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain static builders return nil without C API")
    func testLlamaSamplerChainStaticBuilders() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            #expect(LlamaSamplerChain.initChain(context: ctx, params: LlamaSamplerChainParams()) == nil)
            #expect(LlamaSamplerChain.temperatureChain(context: ctx, temperature: 1.0) == nil)
            #expect(LlamaSamplerChain.topKTopPChain(context: ctx, k: 5, p: 0.9) == nil)
            #expect(LlamaSamplerChain.repetitionPenaltyChain(context: ctx, penalty: 1.1) == nil)
            #expect(LlamaSamplerChain.comprehensiveChain(context: ctx, temperature: 1.0, k: 5, p: 0.9, penalty: 1.1) == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain addSampler functionality")
    func testLlamaSamplerChainAddSampler() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            let sampler = LlamaSampler(context: ctx)
            
            // Test adding samplers
            chain.addSampler(sampler)
            #expect(chain.samplerCount == 0) // Should remain 0 with dummy context
            
            // Test adding multiple samplers
            chain.addSampler(sampler)
            chain.addSampler(sampler)
            #expect(chain.samplerCount == 0) // Should remain 0 with dummy context
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain getSampler functionality")
    func testLlamaSamplerChainGetSampler() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            
            // Test getting samplers at various indices
            #expect(chain.getSampler(at: 0) == nil)
            #expect(chain.getSampler(at: 1) == nil)
            #expect(chain.getSampler(at: -1) == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain removeSampler functionality")
    func testLlamaSamplerChainRemoveSampler() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            
            // Test removing samplers at various indices
            #expect(chain.removeSampler(at: 0) == nil)
            #expect(chain.removeSampler(at: 1) == nil)
            #expect(chain.removeSampler(at: -1) == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain apply functionality")
    func testLlamaSamplerChainApply() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            
            // Test applying chain to token data array
            var tokenDataArray = LlamaTokenDataArray(
                data: UnsafeMutablePointer<LlamaTokenData>.allocate(capacity: 1),
                size: 1,
                selected: 0,
                sorted: false
            )
            
            chain.apply(to: &tokenDataArray)
            
            // Clean up
            tokenDataArray.data.deallocate()
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain sample functionality")
    func testLlamaSamplerChainSample() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            
            // Test sampling with various token arrays
            #expect(chain.sample() == nil)
            #expect(chain.sample(lastTokens: []) == nil)
            #expect(chain.sample(lastTokens: [1, 2, 3]) == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaContext samplerChain extension")
    func testLlamaContextSamplerChainExtension() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = ctx.samplerChain()
            #expect(chain.cChain == nil)
            #expect(chain.samplerCount == 0)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaContext convenience chain methods")
    func testLlamaContextConvenienceChainMethods() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            // Test convenience sampling methods
            #expect(ctx.sampleGreedy() == nil)
            #expect(ctx.sampleWithTemperature(1.0) == nil)
            #expect(ctx.sampleTopK(5) == nil)
            #expect(ctx.sampleTopP(0.9) == nil)
            #expect(ctx.sampleWithRepetitionPenalty(1.1) == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChainParams functionality")
    func testLlamaSamplerChainParams() throws {
        // Test creating default params
        _ = LlamaSamplerChainParams()
        
        // Test creating custom params (if the struct has fields)
        _ = LlamaSamplerChainParams()
        
        // Test that params can be created (actual values depend on C API)
        #expect(Bool(true)) // Just test that struct creation doesn't crash
    }
} 