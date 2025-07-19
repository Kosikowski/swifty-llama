import Testing
@testable import SLlama

struct SLlamaSamplerTests {
    @Test("SLlamaSampler basic functionality")
    func sLlamaSamplerBasicFunctionality() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let sampler = SLlamaSampler(context: ctx)
            let token = sampler.sample()
            #expect(token == nil, "Sampling with dummy context should return nil")
        }
    }

    @Test("SLlamaSampler factory methods")
    func sLlamaSamplerFactoryMethods() throws {
        _ = TestUtilities.withDummyContext { ctx in
            // Test that all factory methods return non-nil samplers
            #expect(SLlamaSampler.greedy(context: ctx) != nil, "Greedy sampler should be created")
            #expect(SLlamaSampler.temperature(context: ctx, temperature: 0.7) != nil, "Temperature sampler should be created")
            #expect(SLlamaSampler.topK(context: ctx, k: 10) != nil, "Top-k sampler should be created")
            #expect(SLlamaSampler.topP(context: ctx, p: 0.9) != nil, "Top-p sampler should be created")
            #expect(SLlamaSampler.repetitionPenalty(context: ctx, penalty: 1.1) != nil, "Repetition penalty sampler should be created")
        }
    }

    @Test("SLlamaSamplerChain initialization and operations")
    func sLlamaSamplerChainInitAndOperations() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let chain = SLlamaSamplerChain(context: ctx)
            let initialized = chain.initialize()
            #expect(initialized == true, "Chain should initialize successfully")
            #expect(chain.samplerCount == 0, "Empty chain should have 0 samplers")

            // Test adding samplers
            if let tempSampler = SLlamaSampler.temperature(context: ctx, temperature: 0.7) {
                chain.addSampler(tempSampler)
                #expect(chain.samplerCount == 1, "Chain should have 1 sampler after adding")
            }

            if let topKSampler = SLlamaSampler.topK(context: ctx, k: 10) {
                chain.addSampler(topKSampler)
                #expect(chain.samplerCount == 2, "Chain should have 2 samplers after adding")
            }

            let token = chain.sample()
            #expect(token == nil, "Sampling with dummy context should return nil")
        }
    }

    @Test("SLlamaSamplerChain cloning")
    func sLlamaSamplerChainCloning() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let chain = SLlamaSamplerChain(context: ctx)
            _ = chain.initialize()

            // Add a sampler to the original chain
            if let tempSampler = SLlamaSampler.temperature(context: ctx, temperature: 0.7) {
                chain.addSampler(tempSampler)
            }

            let clonedChain = chain.clone()
            #expect(clonedChain != nil, "Chain should clone successfully")
            #expect(clonedChain?.samplerCount == chain.samplerCount, "Cloned chain should have same sampler count")
        }
    }

    @Test("SLlamaSamplerChain static builders")
    func sLlamaSamplerChainStaticBuilders() throws {
        _ = TestUtilities.withDummyContext { ctx in
            // Test that all static builders return non-nil chains
            #expect(SLlamaSamplerChain.temperature(context: ctx, temperature: 1.0) != nil, "Temperature chain should be created")
            #expect(SLlamaSamplerChain.topK(context: ctx, k: 5) != nil, "Top-k chain should be created")
            #expect(SLlamaSamplerChain.topP(context: ctx, p: 0.9) != nil, "Top-p chain should be created")
            #expect(SLlamaSamplerChain.repetitionPenalty(context: ctx, penalty: 1.1) != nil, "Repetition penalty chain should be created")
            #expect(SLlamaSamplerChain.custom(context: ctx) != nil, "Custom chain should be created")
        }
    }
}
