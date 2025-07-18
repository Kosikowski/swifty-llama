import Foundation
import llama

/// A wrapper for llama sampling operations
public class SLlamaSampler {
    internal var sampler: SLlamaSamplerPointer?
    private let context: SLlamaContext
    
    /// Initialize with a context
    /// - Parameter context: The llama context to use for sampling
    public init(context: SLlamaContext) {
        self.context = context
    }
    
    deinit {
        if let sampler = sampler {
            llama_sampler_free(sampler)
        }
    }
    
    /// Get the underlying C sampler pointer for direct API access
    public var cSampler: SLlamaSamplerPointer? {
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
    public func accept(_ token: SLlamaToken) {
        guard let sampler = sampler else { return }
        llama_sampler_accept(sampler, token)
    }
    
    /// Apply the sampler to token data array
    /// - Parameter tokenDataArray: The token data array to apply sampling to
    public func apply(to tokenDataArray: SLlamaTokenDataArrayPointer) {
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
    public func clone() -> SLlamaSampler? {
        guard let sampler = sampler else { return nil }
        guard let clonedSampler = llama_sampler_clone(sampler) else { return nil }
        
        let newSampler = SLlamaSampler(context: context)
        newSampler.sampler = clonedSampler
        return newSampler
    }
    
    /// Sample a token from the context
    /// - Parameter lastTokens: Array of last tokens (negative indices for reverse order)
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sample(lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }
        
        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }
        
        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0
        
        if vocabSize == 0 { return nil }
        
        // Create token data array
        var candidates = [SLlamaTokenData]()
        for i in 0..<vocabSize {
            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: logits[Int(i)],
                p: 0.0
            ))
        }
        
        var tokenDataArray = SLlamaTokenDataArray(
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
        
        return SLlamaToken(maxIndex)
    }
    
    /// Sample with temperature
    /// - Parameters:
    ///   - temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
    ///   - lastTokens: Array of last tokens
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleWithTemperature(_ temperature: Float, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }
        
        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }
        
        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0
        
        if vocabSize == 0 { return nil }
        
        // Create token data array with temperature scaling
        var candidates = [SLlamaTokenData]()
        for i in 0..<vocabSize {
            let scaledLogit = temperature > 0 ? logits[Int(i)] / temperature : logits[Int(i)]
            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: scaledLogit,
                p: 0.0
            ))
        }
        
        var tokenDataArray = SLlamaTokenDataArray(
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
        
        return SLlamaToken(maxIndex)
    }
    
    /// Sample with top-k filtering
    /// - Parameters:
    ///   - k: Number of top tokens to consider
    ///   - lastTokens: Array of last tokens
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleTopK(_ k: Int, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }
        
        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }
        
        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0
        
        if vocabSize == 0 { return nil }
        
        // Create token data array
        var candidates = [SLlamaTokenData]()
        for i in 0..<vocabSize {
            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: logits[Int(i)],
                p: 0.0
            ))
        }
        
        // Sort by logit value and take top-k
        candidates.sort { $0.logit > $1.logit }
        candidates = Array(candidates.prefix(k))
        
        var tokenDataArray = SLlamaTokenDataArray(
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
    public func sampleTopP(_ p: Float, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }
        
        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }
        
        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0
        
        if vocabSize == 0 { return nil }
        
        // Create token data array
        var candidates = [SLlamaTokenData]()
        for i in 0..<vocabSize {
            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: logits[Int(i)],
                p: 0.0
            ))
        }
        
        // Sort by logit value
        candidates.sort { $0.logit > $1.logit }
        
        // Calculate cumulative probabilities
        var cumulativeProb = 0.0
        var selectedCandidates: [SLlamaTokenData] = []
        
        for candidate in candidates {
            let prob = exp(candidate.logit)
            cumulativeProb += prob
            selectedCandidates.append(candidate)
            
            if cumulativeProb >= p {
                break
            }
        }
        
        var tokenDataArray = SLlamaTokenDataArray(
            data: selectedCandidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: selectedCandidates.count,
            selected: 0,
            sorted: true
        )
        
        // Apply sampling
        apply(to: &tokenDataArray)
        
        // Find the token with highest probability
        guard let maxIndex = selectedCandidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }
        
        return selectedCandidates[maxIndex].id
    }
    
    /// Sample with repetition penalty
    /// - Parameters:
    ///   - penalty: Repetition penalty factor (1.0 = no penalty, >1.0 = penalty)
    ///   - lastTokens: Array of last tokens to penalize
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleWithRepetitionPenalty(_ penalty: Float, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }
        
        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }
        
        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0
        
        if vocabSize == 0 { return nil }
        
        // Create token data array with repetition penalty
        var candidates = [SLlamaTokenData]()
        for i in 0..<vocabSize {
            var logit = logits[Int(i)]
            
            // Apply repetition penalty
            for lastToken in lastTokens {
                if SLlamaToken(i) == lastToken {
                    logit *= penalty
                    break
                }
            }
            
            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: logit,
                p: 0.0
            ))
        }
        
        var tokenDataArray = SLlamaTokenDataArray(
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
        
        return SLlamaToken(maxIndex)
    }
}

// MARK: - Built-in Sampler Initializers

public extension SLlamaSampler {
    
    /// Initialize a greedy sampler (always picks the highest probability token)
    /// - Returns: A new greedy sampler instance
    static func greedy(context: SLlamaContext) -> SLlamaSampler? {
        guard let samplerPtr = llama_sampler_init_greedy() else { return nil }
        
        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }
    
    /// Initialize a distribution sampler with seed
    /// - Parameters:
    ///   - context: The llama context
    ///   - seed: Random seed for the sampler
    /// - Returns: A new distribution sampler instance
    static func distribution(context: SLlamaContext, seed: UInt32) -> SLlamaSampler? {
        guard let samplerPtr = llama_sampler_init_dist(seed) else { return nil }
        
        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }
}

// MARK: - Extension to SLlamaContext for Sampling

public extension SLlamaContext {
    
    /// Create a sampler for this context
    /// - Returns: A SLlamaSampler instance
    func sampler() -> SLlamaSampler {
        return SLlamaSampler(context: self)
    }
    
    /// Sample a token using greedy sampling
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleGreedy() -> SLlamaToken? {
        guard let sampler = SLlamaSampler.greedy(context: self) else { return nil }
        return sampler.sample()
    }
    
    /// Sample a token using distribution sampling
    /// - Parameter seed: Random seed for the sampler
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleDistribution(seed: UInt32) -> SLlamaToken? {
        guard let sampler = SLlamaSampler.distribution(context: self, seed: seed) else { return nil }
        return sampler.sample()
    }
    
    /// Sample a token with temperature
    /// - Parameter temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithTemperature(_ temperature: Float) -> SLlamaToken? {
        return sampler().sampleWithTemperature(temperature)
    }
    
    /// Sample a token with top-k filtering
    /// - Parameter k: Number of top tokens to consider
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleTopK(_ k: Int) -> SLlamaToken? {
        return sampler().sampleTopK(k)
    }
    
    /// Sample a token with top-p (nucleus) filtering
    /// - Parameter p: Cumulative probability threshold (0.0 to 1.0)
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleTopP(_ p: Float) -> SLlamaToken? {
        return sampler().sampleTopP(p)
    }
    
    /// Sample a token with repetition penalty
    /// - Parameters:
    ///   - penalty: Repetition penalty factor (1.0 = no penalty, higher = more penalty)
    ///   - lastTokens: Array of last tokens to penalize
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithRepetitionPenalty(_ penalty: Float, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        return sampler().sampleWithRepetitionPenalty(penalty, lastTokens: lastTokens)
    }
} 