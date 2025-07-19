import Foundation
import llama

// MARK: - SLlamaSamplerChain

/// A wrapper for llama sampler chain operations
public class SLlamaSamplerChain {
    // MARK: Properties

    private var chain: SLlamaSamplerPointer?
    private let context: SLlamaContext

    // MARK: Computed Properties

    /// Get the underlying C sampler chain pointer for direct API access
    public var cChain: SLlamaSamplerPointer? {
        chain
    }

    /// Get the number of samplers in the chain
    /// - Returns: The number of samplers in the chain
    public var samplerCount: Int32 {
        guard let chain else { return 0 }
        return llama_sampler_chain_n(chain)
    }

    // MARK: Lifecycle

    /// Initialize with a context
    /// - Parameter context: The llama context to use for sampling
    public init(context: SLlamaContext) {
        self.context = context
    }

    deinit {
        if let chain {
            llama_sampler_free(chain)
        }
    }

    // MARK: Functions

    /// Initialize the sampler chain with default parameters
    /// - Returns: true if initialization was successful, false otherwise
    public func initialize() -> Bool {
        let params = SLlamaSamplerChainParams()
        chain = llama_sampler_chain_init(params)
        return chain != nil
    }

    /// Add a sampler to the chain
    /// - Parameter sampler: The sampler to add to the chain
    public func addSampler(_ sampler: SLlamaSampler) {
        guard let chain, let samplerPtr = sampler.cSampler else { return }
        llama_sampler_chain_add(chain, samplerPtr)
    }

    /// Get a sampler from the chain by index
    /// - Parameter index: The index of the sampler to get
    /// - Returns: The sampler at the specified index, or nil if not found
    public func getSampler(at index: Int32) -> SLlamaSampler? {
        guard let chain else { return nil }
        guard let samplerPtr = llama_sampler_chain_get(chain, index) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Remove a sampler from the chain by index
    /// - Parameter index: The index of the sampler to remove
    /// - Returns: The removed sampler, or nil if not found
    public func removeSampler(at index: Int32) -> SLlamaSampler? {
        guard let chain else { return nil }
        guard let samplerPtr = llama_sampler_chain_remove(chain, index) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Sample a token using the entire chain
    ///
    /// **ARCHITECTURAL DECISION**: This method doesn't accept lastTokens parameter because
    /// sampler chains use stateful samplers that maintain their own internal state.
    ///
    /// **How sampler chains work**:
    /// - Each sampler in the chain processes the token data array sequentially
    /// - Penalty samplers in the chain automatically track token history via `accept(token)`
    /// - No manual token tracking needed at the chain level
    /// - Chain.accept(token) propagates to all samplers in the chain
    ///
    /// **Example usage**:
    /// ```swift
    /// let chain = SLlamaSamplerChain.custom(
    ///     context: context,
    ///     temperature: 0.8,
    ///     topK: 40,
    ///     topP: 0.9,
    ///     repetitionPenalty: 1.1
    /// )
    ///
    /// // The chain automatically handles repetition penalties
    /// let token = chain.sample()
    /// chain.accept(token)  // Updates internal state of all penalty samplers
    /// ```
    ///
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sample() -> SLlamaToken? {
        guard let chain else { return nil }
        guard let ctx = context.pointer else { return nil }

        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }

        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0

        if vocabSize == 0 { return nil }

        // Create token data array
        var candidates = [SLlamaTokenData]()
        for i in 0 ..< vocabSize {
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

        // Apply the entire chain - each sampler processes the array in sequence
        // **IMPLEMENTATION NOTE**: This calls each sampler's apply() method in order:
        // 1. Penalty samplers modify logits based on their internal token history
        // 2. Temperature/top-k/top-p samplers filter/scale the candidates
        // 3. Final sampler (e.g., greedy, dist) makes the selection
        llama_sampler_apply(chain, &tokenDataArray)

        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }

        return SLlamaToken(maxIndex)
    }

    /// Accept a token (updates internal state of samplers in the chain)
    /// - Parameter token: The token to accept
    public func accept(_ token: SLlamaToken) {
        guard let chain else { return }
        llama_sampler_accept(chain, token)
    }

    /// Reset the chain state
    public func reset() {
        guard let chain else { return }
        llama_sampler_reset(chain)
    }

    /// Clone the sampler chain
    /// - Returns: A new sampler chain instance, or nil if cloning failed
    public func clone() -> SLlamaSamplerChain? {
        guard let chain else { return nil }
        guard let clonedChain = llama_sampler_clone(chain) else { return nil }

        let newChain = SLlamaSamplerChain(context: context)
        newChain.chain = clonedChain
        return newChain
    }
}

// MARK: - Built-in Chain Configurations

public extension SLlamaSamplerChain {
    /// Create a temperature-based sampling chain
    /// - Parameters:
    ///   - context: The llama context
    ///   - temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
    /// - Returns: A configured sampler chain, or nil if initialization failed
    static func temperature(
        context: SLlamaContext,
        temperature: Float = 0.7
    )
        -> SLlamaSamplerChain?
    {
        let chain = SLlamaSamplerChain(context: context)
        guard chain.initialize() else { return nil }

        // Add temperature sampler
        guard let tempSampler = SLlamaSampler.temperature(context: context, temperature: temperature) else {
            return nil
        }
        chain.addSampler(tempSampler)

        return chain
    }

    /// Create a top-k sampling chain
    /// - Parameters:
    ///   - context: The llama context
    ///   - k: Number of top tokens to consider
    /// - Returns: A configured sampler chain, or nil if initialization failed
    static func topK(
        context: SLlamaContext,
        k: Int32 = 40
    )
        -> SLlamaSamplerChain?
    {
        let chain = SLlamaSamplerChain(context: context)
        guard chain.initialize() else { return nil }

        // Add top-k sampler
        guard let topKSampler = SLlamaSampler.topK(context: context, k: k) else {
            return nil
        }
        chain.addSampler(topKSampler)

        return chain
    }

    /// Create a top-p (nucleus) sampling chain
    /// - Parameters:
    ///   - context: The llama context
    ///   - p: Cumulative probability threshold (0.0 to 1.0)
    /// - Returns: A configured sampler chain, or nil if initialization failed
    static func topP(
        context: SLlamaContext,
        p: Float = 0.9
    )
        -> SLlamaSamplerChain?
    {
        let chain = SLlamaSamplerChain(context: context)
        guard chain.initialize() else { return nil }

        // Add top-p sampler
        guard let topPSampler = SLlamaSampler.topP(context: context, p: p) else {
            return nil
        }
        chain.addSampler(topPSampler)

        return chain
    }

    /// Create a repetition penalty sampling chain
    /// - Parameters:
    ///   - context: The llama context
    ///   - penalty: Repetition penalty factor (1.0 = no penalty, >1.0 = penalty)
    /// - Returns: A configured sampler chain, or nil if initialization failed
    static func repetitionPenalty(
        context: SLlamaContext,
        penalty: Float = 1.1
    )
        -> SLlamaSamplerChain?
    {
        let chain = SLlamaSamplerChain(context: context)
        guard chain.initialize() else { return nil }

        // Add repetition penalty sampler
        guard let penaltySampler = SLlamaSampler.repetitionPenalty(context: context, penalty: penalty) else {
            return nil
        }
        chain.addSampler(penaltySampler)

        return chain
    }

    /// Create a custom sampling chain with multiple strategies
    ///
    /// **ARCHITECTURAL DECISION**: The maxTokens parameter was removed because sampler chains
    /// define HOW to sample (temperature, penalties, filtering), not HOW MANY tokens to generate.
    ///
    /// **Why maxTokens doesn't belong here**:
    /// - Sampler chains are reusable sampling strategies, not generation loops
    /// - Token count limits belong in the generation loop, not the sampler configuration
    /// - Separation of concerns: samplers handle probability, generators handle quantity
    /// - A single chain can be used for different generation lengths
    ///
    /// **Where to control token count**:
    /// ```swift
    /// let chain = SLlamaSamplerChain.custom(
    ///     context: context,
    ///     temperature: 0.8,
    ///     topK: 40,
    ///     topP: 0.9,
    ///     repetitionPenalty: 1.1
    /// )
    ///
    /// // ✅ Control token count in the generation loop
    /// var tokens: [SLlamaToken] = []
    /// let maxTokens = 1000
    ///
    /// for _ in 0..<maxTokens {
    ///     guard let token = chain.sample() else { break }
    ///     tokens.append(token)
    ///     chain.accept(token)
    ///
    ///     // Add your stopping conditions here
    ///     if token == eosToken { break }
    /// }
    /// ```
    ///
    /// **Chain composition strategy**:
    /// This method creates a chain with the most commonly used sampling techniques:
    /// 1. **Repetition penalty** - Reduces repetition by penalizing recently used tokens
    /// 2. **Top-k filtering** - Limits vocabulary to k most likely tokens
    /// 3. **Top-p filtering** - Further filters by cumulative probability
    /// 4. **Temperature scaling** - Controls randomness of final selection
    ///
    /// - Parameters:
    ///   - context: The llama context
    ///   - temperature: Temperature for sampling
    ///   - topK: Number of top tokens to consider
    ///   - topP: Cumulative probability threshold
    ///   - repetitionPenalty: Repetition penalty factor
    /// - Returns: A configured sampler chain, or nil if initialization failed
    static func custom(
        context: SLlamaContext,
        temperature: Float = 0.7,
        topK: Int32 = 40,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.1
    )
        -> SLlamaSamplerChain?
    {
        let chain = SLlamaSamplerChain(context: context)
        guard chain.initialize() else { return nil }

        // **ORDER MATTERS**: Samplers are applied in the order they're added

        // 1. Add repetition penalty first - modifies logits based on token history
        if let penaltySampler = SLlamaSampler.repetitionPenalty(context: context, penalty: repetitionPenalty) {
            chain.addSampler(penaltySampler)
        }

        // 2. Add top-k sampler - reduces vocabulary to k most likely tokens
        if let topKSampler = SLlamaSampler.topK(context: context, k: topK) {
            chain.addSampler(topKSampler)
        }

        // 3. Add top-p sampler - further filters by cumulative probability
        if let topPSampler = SLlamaSampler.topP(context: context, p: topP) {
            chain.addSampler(topPSampler)
        }

        // 4. Add temperature sampler - controls randomness of final selection
        if let tempSampler = SLlamaSampler.temperature(context: context, temperature: temperature) {
            chain.addSampler(tempSampler)
        }

        return chain
    }
}

// MARK: - Extension to SLlamaContext for Chain Sampling

public extension SLlamaContext {
    /// Create a sampler chain for this context
    /// - Returns: A SLlamaSamplerChain instance
    func samplerChain() -> SLlamaSamplerChain {
        SLlamaSamplerChain(context: self)
    }

    /// Sample a token using a temperature-based chain
    /// - Parameter temperature: Temperature for sampling
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithTemperatureChain(_ temperature: Float = 0.7) -> SLlamaToken? {
        guard let chain = SLlamaSamplerChain.temperature(context: self, temperature: temperature) else {
            return nil
        }
        return chain.sample()
    }

    /// Sample a token using a top-k chain
    /// - Parameter k: Number of top tokens to consider
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithTopKChain(_ k: Int32 = 40) -> SLlamaToken? {
        guard let chain = SLlamaSamplerChain.topK(context: self, k: k) else {
            return nil
        }
        return chain.sample()
    }

    /// Sample a token using a top-p chain
    /// - Parameter p: Cumulative probability threshold
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithTopPChain(_ p: Float = 0.9) -> SLlamaToken? {
        guard let chain = SLlamaSamplerChain.topP(context: self, p: p) else {
            return nil
        }
        return chain.sample()
    }

    /// Sample a token using a custom chain
    /// - Parameters:
    ///   - temperature: Temperature for sampling
    ///   - topK: Number of top tokens to consider
    ///   - topP: Cumulative probability threshold
    ///   - repetitionPenalty: Repetition penalty factor
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithCustomChain(
        temperature: Float = 0.7,
        topK: Int32 = 40,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.1
    )
        -> SLlamaToken?
    {
        guard
            let chain = SLlamaSamplerChain.custom(
                context: self,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty
            )
        else {
            return nil
        }
        return chain.sample()
    }
}
