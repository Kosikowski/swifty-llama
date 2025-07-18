import Testing
import SwiftyLlamaCpp

@testable import SwiftyLlamaCpp

struct SLlamaSamplerTests {
    
    @Test("SLlamaSampler initialization")
    func testSLlamaSamplerInit() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let sampler = SLlamaSampler(context: ctx)
            #expect(sampler != nil, "Sampler should be created successfully")
        }
    }
    
    @Test("SLlamaSampler basic sampling")
    func testSLlamaSamplerBasicSampling() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let sampler = SLlamaSampler(context: ctx)
            let token = sampler.sample()
            #expect(token == nil, "Sampling with dummy context should return nil")
        }
    }
    
    @Test("SLlamaSampler greedy sampling")
    func testSLlamaSamplerGreedySampling() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let sampler = SLlamaSampler.greedy(context: ctx)
            #expect(sampler != nil, "Greedy sampler should be created successfully")
            let token = sampler?.sample()
            #expect(token == nil, "Greedy sampling with dummy context should return nil")
        }
    }
    
    @Test("SLlamaSampler temperature sampling")
    func testSLlamaSamplerTemperatureSampling() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let sampler = SLlamaSampler.temperature(context: ctx, temperature: 0.7)
            #expect(sampler != nil, "Temperature sampler should be created successfully")
            let token = sampler?.sample()
            #expect(token == nil, "Temperature sampling with dummy context should return nil")
        }
    }
    
    @Test("SLlamaSampler top-k sampling")
    func testSLlamaSamplerTopKSampling() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let sampler = SLlamaSampler.topK(context: ctx, k: 10)
            #expect(sampler != nil, "Top-k sampler should be created successfully")
            let token = sampler?.sample()
            #expect(token == nil, "Top-k sampling with dummy context should return nil")
        }
    }
    
    @Test("SLlamaSampler top-p sampling")
    func testSLlamaSamplerTopPSampling() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let sampler = SLlamaSampler.topP(context: ctx, p: 0.9)
            #expect(sampler != nil, "Top-p sampler should be created successfully")
            let token = sampler?.sample()
            #expect(token == nil, "Top-p sampling with dummy context should return nil")
        }
    }
    
    @Test("SLlamaSampler repetition penalty sampling")
    func testSLlamaSamplerRepetitionPenaltySampling() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let sampler = SLlamaSampler.repetitionPenalty(context: ctx, penalty: 1.1)
            #expect(sampler != nil, "Repetition penalty sampler should be created successfully")
            let token = sampler?.sample()
            #expect(token == nil, "Repetition penalty sampling with dummy context should return nil")
        }
    }
    
    @Test("SLlamaSamplerChain initialization")
    func testSLlamaSamplerChainInit() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let chain = SLlamaSamplerChain(context: ctx)
            #expect(chain != nil, "Sampler chain should be created successfully")
        }
    }
    
    @Test("SLlamaSamplerChain static builders")
    func testSLlamaSamplerChainStaticBuilders() throws {
        _ = TestUtilities.withDummyContext { ctx in
            #expect(SLlamaSamplerChain.temperature(context: ctx, temperature: 1.0) != nil)
            #expect(SLlamaSamplerChain.topK(context: ctx, k: 5) != nil)
            #expect(SLlamaSamplerChain.topP(context: ctx, p: 0.9) != nil)
            #expect(SLlamaSamplerChain.repetitionPenalty(context: ctx, penalty: 1.1) != nil)
            #expect(SLlamaSamplerChain.custom(context: ctx) != nil)
        }
    }
    
    @Test("SLlamaSamplerChain operations")
    func testSLlamaSamplerChainOperations() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let chain = SLlamaSamplerChain(context: ctx)
            let initialized = chain.initialize()
            #expect(initialized == true, "Chain should initialize successfully")
            #expect(chain.samplerCount == 0, "Empty chain should have 0 samplers")
            if let tempSampler = SLlamaSampler.temperature(context: ctx, temperature: 0.7) {
                chain.addSampler(tempSampler)
                #expect(chain.samplerCount == 1, "Chain should have 1 sampler after adding")
            }
            let token = chain.sample()
            #expect(token == nil, "Sampling with dummy context should return nil")
        }
    }
    
    @Test("SLlamaSamplerChain cloning")
    func testSLlamaSamplerChainCloning() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let chain = SLlamaSamplerChain(context: ctx)
            chain.initialize()
            if let tempSampler = SLlamaSampler.temperature(context: ctx, temperature: 0.7) {
                chain.addSampler(tempSampler)
            }
            let clonedChain = chain.clone()
            #expect(clonedChain != nil, "Chain should clone successfully")
        }
    }
    
    @Test("SLlamaSamplerChain token data array operations")
    func testSLlamaSamplerChainTokenDataArrayOperations() throws {
        _ = TestUtilities.withDummyContext { ctx in
            let chain = SLlamaSamplerChain(context: ctx)
            chain.initialize()
            // Create a simple token data array
            var candidates = [
                SLlamaTokenData(id: 1, logit: 0.5, p: 0.0),
                SLlamaTokenData(id: 2, logit: 0.3, p: 0.0),
                SLlamaTokenData(id: 3, logit: 0.2, p: 0.0)
            ]
            let tokenDataArray = SLlamaTokenDataArray(
                data: candidates.withUnsafeMutableBufferPointer { $0.baseAddress },
                size: candidates.count,
                selected: 0,
                sorted: false
            )
            #expect(tokenDataArray.size == 3, "Token data array should have 3 elements")
            #expect(tokenDataArray.selected == 0, "Token data array should start with selected = 0")
            #expect(tokenDataArray.sorted == false, "Token data array should start with sorted = false")
        }
    }
} 