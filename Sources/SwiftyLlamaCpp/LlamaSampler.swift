import Foundation
import llama

/// A wrapper for llama sampling operations
public class LlamaSampler {
    internal var sampler: LlamaSamplerPointer?
    private let context: LlamaContext
    
    /// Initialize with a context
    /// - Parameter context: The llama context to use for sampling
    public init(context: LlamaContext) {
        self.context = context
    }
    
    deinit {
        if let sampler = sampler {
            llama_sampler_free(sampler)
        }
    }
    
    /// Get the underlying C sampler pointer for direct API access
    public var cSampler: LlamaSamplerPointer? {
        return sampler
    }
    
    /// Get the sampler name
    /// - Returns: The name of the sampler, or nil if not available
    public var name: String? {
        guard let sampler = sampler else { return nil }
        return String(cString: llama_sampler_name(sampler))
    }
    
    /// Accept a token (updates internal state of certain samplers)
    /// - Parameter token: The token to accept
    public func accept(_ token: LlamaToken) {
        guard let sampler = sampler else { return }
        llama_sampler_accept(sampler, token)
    }
    
    /// Apply the sampler to token data array
    /// - Parameter tokenDataArray: The token data array to apply sampling to
    public func apply(to tokenDataArray: LlamaTokenDataArrayPointer) {
        guard let sampler = sampler else { return }
        llama_sampler_apply(sampler, tokenDataArray)
    }
    
    /// Reset the sampler state
    public func reset() {
        guard let sampler = sampler else { return }
        llama_sampler_reset(sampler)
    }
    
    /// Clone the sampler
    /// - Returns: A new sampler instance, or nil if cloning failed
    public func clone() -> LlamaSampler? {
        guard let sampler = sampler else { return nil }
        guard let clonedSampler = llama_sampler_clone(sampler) else { return nil }
        
        let newSampler = LlamaSampler(context: context)
        newSampler.sampler = clonedSampler
        return newSampler
    }
    
    /// Sample a token from the context
    /// - Parameter lastTokens: Array of last tokens (negative indices for reverse order)
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
        
        // Apply sampling
        apply(to: &tokenDataArray)
        
        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }
        
        return LlamaToken(maxIndex)
    }
    
    /// Sample with temperature
    /// - Parameters:
    ///   - temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
    ///   - lastTokens: Array of last tokens
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleWithTemperature(_ temperature: Float, lastTokens: [LlamaToken] = []) -> LlamaToken? {
        guard let ctx = context.pointer else { return nil }
        
        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }
        
        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0
        
        if vocabSize == 0 { return nil }
        
        // Create token data array with temperature scaling
        var candidates = [LlamaTokenData]()
        for i in 0..<vocabSize {
            let scaledLogit = temperature > 0 ? logits[Int(i)] / temperature : logits[Int(i)]
            candidates.append(LlamaTokenData(
                id: LlamaToken(i),
                logit: scaledLogit,
                p: 0.0
            ))
        }
        
        var tokenDataArray = LlamaTokenDataArray(
            data: candidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: candidates.count,
            selected: 0,
            sorted: false
        )
        
        // Apply sampling
        apply(to: &tokenDataArray)
        
        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }
        
        return LlamaToken(maxIndex)
    }
    
    /// Sample with top-k filtering
    /// - Parameters:
    ///   - k: Number of top tokens to consider
    ///   - lastTokens: Array of last tokens
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleTopK(_ k: Int, lastTokens: [LlamaToken] = []) -> LlamaToken? {
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
        
        // Sort by logit value and take top-k
        candidates.sort { $0.logit > $1.logit }
        candidates = Array(candidates.prefix(k))
        
        var tokenDataArray = LlamaTokenDataArray(
            data: candidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: candidates.count,
            selected: 0,
            sorted: true
        )
        
        // Apply sampling
        apply(to: &tokenDataArray)
        
        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }
        
        return candidates[maxIndex].id
    }
    
    /// Sample with top-p (nucleus) filtering
    /// - Parameters:
    ///   - p: Cumulative probability threshold (0.0 to 1.0)
    ///   - lastTokens: Array of last tokens
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleTopP(_ p: Float, lastTokens: [LlamaToken] = []) -> LlamaToken? {
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
        
        // Sort by logit value
        candidates.sort { $0.logit > $1.logit }
        
        // Apply softmax to get probabilities
        let maxLogit = candidates.first?.logit ?? 0
        let expLogits = candidates.map { exp($0.logit - maxLogit) }
        let sumExp = expLogits.reduce(0, +)
        let probabilities = expLogits.map { $0 / sumExp }
        
        // Apply top-p filtering
        var cumulativeProb = Float(0)
        var filteredCandidates = [LlamaTokenData]()
        
        for (index, prob) in probabilities.enumerated() {
            cumulativeProb += prob
            filteredCandidates.append(LlamaTokenData(
                id: candidates[index].id,
                logit: candidates[index].logit,
                p: prob
            ))
            
            if cumulativeProb >= p {
                break
            }
        }
        
        var tokenDataArray = LlamaTokenDataArray(
            data: filteredCandidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: filteredCandidates.count,
            selected: 0,
            sorted: false
        )
        
        // Apply sampling
        apply(to: &tokenDataArray)
        
        // Find the token with highest probability
        guard let maxIndex = filteredCandidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }
        
        return filteredCandidates[maxIndex].id
    }
    
    /// Sample with repetition penalty
    /// - Parameters:
    ///   - penalty: Repetition penalty factor (1.0 = no penalty, higher = more penalty)
    ///   - lastTokens: Array of last tokens to penalize
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleWithRepetitionPenalty(_ penalty: Float, lastTokens: [LlamaToken] = []) -> LlamaToken? {
        guard let ctx = context.pointer else { return nil }
        
        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }
        
        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0
        
        if vocabSize == 0 { return nil }
        
        // Create token data array with repetition penalty
        var candidates = [LlamaTokenData]()
        for i in 0..<vocabSize {
            var logit = logits[Int(i)]
            
            // Apply repetition penalty
            for token in lastTokens {
                if token == LlamaToken(i) {
                    logit *= penalty
                    break
                }
            }
            
            candidates.append(LlamaTokenData(
                id: LlamaToken(i),
                logit: logit,
                p: 0.0
            ))
        }
        
        var tokenDataArray = LlamaTokenDataArray(
            data: candidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: candidates.count,
            selected: 0,
            sorted: false
        )
        
        // Apply sampling
        apply(to: &tokenDataArray)
        
        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }
        
        return LlamaToken(maxIndex)
    }
}

// MARK: - Built-in Sampler Initializers

public extension LlamaSampler {
    
    /// Initialize a greedy sampler (always picks the highest probability token)
    /// - Returns: A new greedy sampler instance
    static func greedy(context: LlamaContext) -> LlamaSampler? {
        guard let samplerPtr = llama_sampler_init_greedy() else { return nil }
        
        let sampler = LlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }
    
    /// Initialize a distribution sampler with seed
    /// - Parameters:
    ///   - context: The llama context
    ///   - seed: Random seed for the sampler
    /// - Returns: A new distribution sampler instance
    static func distribution(context: LlamaContext, seed: UInt32) -> LlamaSampler? {
        guard let samplerPtr = llama_sampler_init_dist(seed) else { return nil }
        
        let sampler = LlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }
}

// MARK: - Extension to LlamaContext for Sampling

public extension LlamaContext {
    
    /// Create a sampler for this context
    /// - Returns: A LlamaSampler instance
    func sampler() -> LlamaSampler {
        return LlamaSampler(context: self)
    }
    
    /// Sample a token using greedy sampling
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleGreedy() -> LlamaToken? {
        guard let sampler = LlamaSampler.greedy(context: self) else { return nil }
        return sampler.sample()
    }
    
    /// Sample a token using distribution sampling
    /// - Parameter seed: Random seed for the sampler
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleDistribution(seed: UInt32) -> LlamaToken? {
        guard let sampler = LlamaSampler.distribution(context: self, seed: seed) else { return nil }
        return sampler.sample()
    }
    
    /// Sample a token with temperature
    /// - Parameter temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithTemperature(_ temperature: Float) -> LlamaToken? {
        return sampler().sampleWithTemperature(temperature)
    }
    
    /// Sample a token with top-k filtering
    /// - Parameter k: Number of top tokens to consider
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleTopK(_ k: Int) -> LlamaToken? {
        return sampler().sampleTopK(k)
    }
    
    /// Sample a token with top-p (nucleus) filtering
    /// - Parameter p: Cumulative probability threshold (0.0 to 1.0)
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleTopP(_ p: Float) -> LlamaToken? {
        return sampler().sampleTopP(p)
    }
    
    /// Sample a token with repetition penalty
    /// - Parameters:
    ///   - penalty: Repetition penalty factor (1.0 = no penalty, higher = more penalty)
    ///   - lastTokens: Array of last tokens to penalize
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithRepetitionPenalty(_ penalty: Float, lastTokens: [LlamaToken] = []) -> LlamaToken? {
        return sampler().sampleWithRepetitionPenalty(penalty, lastTokens: lastTokens)
    }
} 