import Foundation
import llama

/// A wrapper for llama sampler chain operations
public class LlamaSamplerChain {
    private var chain: LlamaSamplerPointer?
    private let context: LlamaContext
    
    /// Initialize with a context
    /// - Parameter context: The llama context to use for sampling
    public init(context: LlamaContext) {
        self.context = context
    }
    
    deinit {
        if let chain = chain {
            llama_sampler_free(chain)
        }
    }
    
    /// Get the underlying C chain pointer for direct API access
    public var cChain: LlamaSamplerPointer? {
        return chain
    }
    
    /// Initialize a sampler chain with parameters
    /// - Parameter params: The sampler chain parameters
    /// - Returns: A new sampler chain instance, or nil if initialization failed
    public static func initChain(context: LlamaContext, params: LlamaSamplerChainParams) -> LlamaSamplerChain? {
        guard let chainPtr = llama_sampler_chain_init(params) else { return nil }
        
        let chain = LlamaSamplerChain(context: context)
        chain.chain = chainPtr
        return chain
    }
    
    /// Add a sampler to the chain
    /// - Parameter sampler: The sampler to add to the chain
    public func addSampler(_ sampler: LlamaSampler) {
        guard let chain = chain, let samplerPtr = sampler.sampler else { return }
        llama_sampler_chain_add(chain, samplerPtr)
    }
    
    /// Get a sampler from the chain by index
    /// - Parameter index: The index of the sampler to get
    /// - Returns: The sampler at the specified index, or nil if index is out of bounds
    public func getSampler(at index: Int32) -> LlamaSampler? {
        guard let chain = chain else { return nil }
        guard let samplerPtr = llama_sampler_chain_get(chain, index) else { return nil }
        
        let sampler = LlamaSampler(context: context)
        sampler.sampler = UnsafeMutablePointer(mutating: samplerPtr)
        return sampler
    }
    
    /// Get the number of samplers in the chain
    /// - Returns: The number of samplers in the chain
    public var samplerCount: Int32 {
        guard let chain = chain else { return 0 }
        return llama_sampler_chain_n(chain)
    }
    
    /// Remove a sampler from the chain by index
    /// - Parameter index: The index of the sampler to remove
    /// - Returns: The removed sampler, or nil if index is out of bounds
    public func removeSampler(at index: Int32) -> LlamaSampler? {
        guard let chain = chain else { return nil }
        guard let samplerPtr = llama_sampler_chain_remove(chain, index) else { return nil }
        
        let sampler = LlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }
    
    /// Apply the entire chain to token data array
    /// - Parameter tokenDataArray: The token data array to apply the chain to
    public func apply(to tokenDataArray: LlamaTokenDataArrayPointer) {
        guard let chain = chain else { return }
        llama_sampler_apply(chain, tokenDataArray)
    }
    
    /// Accept a token through the entire chain
    /// - Parameter token: The token to accept
    public func accept(_ token: LlamaToken) {
        guard let chain = chain else { return }
        llama_sampler_accept(chain, token)
    }
    
    /// Reset the entire chain
    public func reset() {
        guard let chain = chain else { return }
        llama_sampler_reset(chain)
    }
    
    /// Clone the entire chain
    /// - Returns: A new sampler chain instance, or nil if cloning failed
    public func clone() -> LlamaSamplerChain? {
        guard let chain = chain else { return nil }
        guard let clonedChain = llama_sampler_clone(chain) else { return nil }
        
        let newChain = LlamaSamplerChain(context: context)
        newChain.chain = UnsafeMutablePointer(mutating: clonedChain)
        return newChain
    }
    
    /// Sample a token using the entire chain
    /// - Parameter lastTokens: Array of last tokens
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sample(lastTokens: [LlamaToken] = []) -> LlamaToken? {
        guard let ctx = context.pointer else { return nil }
        
        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }
        
        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0
        
        if vocabSize == 0 { return nil }
        
        // Create token data array
        var candidates = [LlamaTokenData]()
        for i in 0..<vocabSize {
            candidates.append(LlamaTokenData(
                id: LlamaToken(i),
                logit: logits[Int(i)],
                p: 0.0
            ))
        }
        
        var tokenDataArray = LlamaTokenDataArray(
            data: candidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: candidates.count,
            selected: 0,
            sorted: false
        )
        
        // Apply the entire chain
        apply(to: &tokenDataArray)
        
        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }
        
        return LlamaToken(maxIndex)
    }
}

// MARK: - Convenience Chain Builders

public extension LlamaSamplerChain {
    
    /// Create a chain with temperature sampling
    /// - Parameters:
    ///   - context: The llama context
    ///   - temperature: Temperature for sampling
    /// - Returns: A sampler chain with temperature sampling
    static func temperatureChain(context: LlamaContext, temperature: Float) -> LlamaSamplerChain? {
        let params = LlamaSamplerChainParams()
        guard let chain = initChain(context: context, params: params) else { return nil }
        
        // Add temperature sampler
        let tempSampler = LlamaSampler(context: context)
        chain.addSampler(tempSampler)
        
        return chain
    }
    
    /// Create a chain with top-k and top-p sampling
    /// - Parameters:
    ///   - context: The llama context
    ///   - k: Number of top tokens to consider
    ///   - p: Cumulative probability threshold
    /// - Returns: A sampler chain with top-k and top-p sampling
    static func topKTopPChain(context: LlamaContext, k: Int, p: Float) -> LlamaSamplerChain? {
        let params = LlamaSamplerChainParams()
        guard let chain = initChain(context: context, params: params) else { return nil }
        
        // Add top-k sampler
        let topKSampler = LlamaSampler(context: context)
        chain.addSampler(topKSampler)
        
        // Add top-p sampler
        let topPSampler = LlamaSampler(context: context)
        chain.addSampler(topPSampler)
        
        return chain
    }
    
    /// Create a chain with repetition penalty
    /// - Parameters:
    ///   - context: The llama context
    ///   - penalty: Repetition penalty factor
    /// - Returns: A sampler chain with repetition penalty
    static func repetitionPenaltyChain(context: LlamaContext, penalty: Float) -> LlamaSamplerChain? {
        let params = LlamaSamplerChainParams()
        guard let chain = initChain(context: context, params: params) else { return nil }
        
        // Add repetition penalty sampler
        let penaltySampler = LlamaSampler(context: context)
        chain.addSampler(penaltySampler)
        
        return chain
    }
    
    /// Create a comprehensive sampling chain
    /// - Parameters:
    ///   - context: The llama context
    ///   - temperature: Temperature for sampling
    ///   - k: Number of top tokens to consider
    ///   - p: Cumulative probability threshold
    ///   - penalty: Repetition penalty factor
    /// - Returns: A comprehensive sampler chain
    static func comprehensiveChain(
        context: LlamaContext,
        temperature: Float,
        k: Int,
        p: Float,
        penalty: Float
    ) -> LlamaSamplerChain? {
        let params = LlamaSamplerChainParams()
        guard let chain = initChain(context: context, params: params) else { return nil }
        
        // Add temperature sampler
        let tempSampler = LlamaSampler(context: context)
        chain.addSampler(tempSampler)
        
        // Add top-k sampler
        let topKSampler = LlamaSampler(context: context)
        chain.addSampler(topKSampler)
        
        // Add top-p sampler
        let topPSampler = LlamaSampler(context: context)
        chain.addSampler(topPSampler)
        
        // Add repetition penalty sampler
        let penaltySampler = LlamaSampler(context: context)
        chain.addSampler(penaltySampler)
        
        return chain
    }
}

// MARK: - Extension to LlamaContext for Sampler Chains

public extension LlamaContext {
    
    /// Create a sampler chain for this context
    /// - Returns: A LlamaSamplerChain instance
    func samplerChain() -> LlamaSamplerChain {
        return LlamaSamplerChain(context: self)
    }
    
    /// Create a temperature sampling chain
    /// - Parameter temperature: Temperature for sampling
    /// - Returns: A sampler chain with temperature sampling
    func temperatureChain(_ temperature: Float) -> LlamaSamplerChain? {
        return LlamaSamplerChain.temperatureChain(context: self, temperature: temperature)
    }
    
    /// Create a top-k and top-p sampling chain
    /// - Parameters:
    ///   - k: Number of top tokens to consider
    ///   - p: Cumulative probability threshold
    /// - Returns: A sampler chain with top-k and top-p sampling
    func topKTopPChain(k: Int, p: Float) -> LlamaSamplerChain? {
        return LlamaSamplerChain.topKTopPChain(context: self, k: k, p: p)
    }
    
    /// Create a repetition penalty chain
    /// - Parameter penalty: Repetition penalty factor
    /// - Returns: A sampler chain with repetition penalty
    func repetitionPenaltyChain(_ penalty: Float) -> LlamaSamplerChain? {
        return LlamaSamplerChain.repetitionPenaltyChain(context: self, penalty: penalty)
    }
    
    /// Create a comprehensive sampling chain
    /// - Parameters:
    ///   - temperature: Temperature for sampling
    ///   - k: Number of top tokens to consider
    ///   - p: Cumulative probability threshold
    ///   - penalty: Repetition penalty factor
    /// - Returns: A comprehensive sampler chain
    func comprehensiveChain(
        temperature: Float,
        k: Int,
        p: Float,
        penalty: Float
    ) -> LlamaSamplerChain? {
        return LlamaSamplerChain.comprehensiveChain(
            context: self,
            temperature: temperature,
            k: k,
            p: p,
            penalty: penalty
        )
    }
    
    /// Sample using a comprehensive chain with default parameters
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleComprehensive() -> LlamaToken? {
        guard let chain = comprehensiveChain(
            temperature: 0.7,
            k: 40,
            p: 0.9,
            penalty: 1.1
        ) else { return nil }
        
        return chain.sample()
    }
}

// MARK: - Sampler Chain Parameters

public extension LlamaSamplerChainParams {
    
    /// Create default sampler chain parameters
    /// - Returns: Default sampler chain parameters
    static func `default`() -> LlamaSamplerChainParams {
        return LlamaSamplerChainParams()
    }
    
    /// Create sampler chain parameters with custom settings
    /// - Parameters:
    ///   - maxTokens: Maximum number of tokens in the chain
    ///   - temperature: Default temperature for sampling
    ///   - topK: Default top-k value
    ///   - topP: Default top-p value
    ///   - repetitionPenalty: Default repetition penalty
    /// - Returns: Custom sampler chain parameters
    static func custom(
        maxTokens: Int32 = 1000,
        temperature: Float = 0.7,
        topK: Int32 = 40,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.1
    ) -> LlamaSamplerChainParams {
        let params = LlamaSamplerChainParams()
        // Note: The actual structure fields would depend on the C API
        // This is a placeholder for the actual implementation
        return params
    }
} 