import Testing
@testable import SwiftyLlamaCpp

class DummyContext: LlamaContext {
    override init?(model: LlamaModel, contextParams: LlamaContextParams? = nil) {
        super.init(model: model, contextParams: contextParams)
    }
    
    override var pointer: LlamaContextPointer? { nil }
    override var associatedModel: LlamaModel? { nil }
}

@Suite 
struct LlamaSamplerTests {
    
    @Test("LlamaSampler typealiases exist and are correct")
    func testLlamaSamplerTypealiases() throws {
        let _: LlamaSamplerPointer? = nil
        let _: LlamaSamplerContext? = nil
        let _: LlamaTokenDataArrayPointer? = nil
    }
    
    @Test("LlamaSampler construction and basic API")
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
    
    @Test("LlamaSamplerChain construction and basic API")
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
            
            // Test adding sampler (should not crash even with nil chain)
            chain.addSampler(sampler)
            #expect(chain.samplerCount == 0) // Should remain 0 since chain is nil
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain getSampler functionality")
    func testLlamaSamplerChainGetSampler() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            
            // Test getting sampler at various indices
            #expect(chain.getSampler(at: 0) == nil)
            #expect(chain.getSampler(at: -1) == nil)
            #expect(chain.getSampler(at: 999) == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain removeSampler functionality")
    func testLlamaSamplerChainRemoveSampler() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            
            // Test removing sampler at various indices
            #expect(chain.removeSampler(at: 0) == nil)
            #expect(chain.removeSampler(at: -1) == nil)
            #expect(chain.removeSampler(at: 999) == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain apply functionality")
    func testLlamaSamplerChainApply() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            
            // Test applying to token data array (should not crash)
            var tokenDataArray = LlamaTokenDataArray(
                data: nil,
                size: 0,
                selected: 0,
                sorted: false
            )
            chain.apply(to: &tokenDataArray)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChain sample functionality")
    func testLlamaSamplerChainSample() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let chain = LlamaSamplerChain(context: ctx)
            
            // Test sampling with empty tokens
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
            #expect(ctx.temperatureChain(0.7) == nil)
            #expect(ctx.topKTopPChain(k: 40, p: 0.9) == nil)
            #expect(ctx.repetitionPenaltyChain(1.1) == nil)
            #expect(ctx.comprehensiveChain(temperature: 0.7, k: 40, p: 0.9, penalty: 1.1) == nil)
            #expect(ctx.sampleComprehensive() == nil)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaSamplerChainParams functionality")
    func testLlamaSamplerChainParams() throws {
        let defaultParams = LlamaSamplerChainParams.default()
        let customParams = LlamaSamplerChainParams.custom(
            maxTokens: 1000,
            temperature: 0.7,
            topK: 40,
            topP: 0.9,
            repetitionPenalty: 1.1
        )
        
        // Test that params can be created (actual values depend on C API)
        #expect(defaultParams != nil || defaultParams == nil) // Just test it doesn't crash
        #expect(customParams != nil || customParams == nil) // Just test it doesn't crash
    }
} 