import Foundation
import llama

// MARK: - SLlamaSampler

/// A wrapper for llama sampling operations
///
/// The implementation is split across multiple files for maintainability:
/// - `SLlamaSampler.swift` - Core class with sampling methods
/// - `SLlamaSampler+Factory.swift` - Static factory methods for various sampler types
/// - `SLlamaContext+Sampling.swift` - Convenience sampling methods on SLlamaContext
///
/// Thread Safety: This class wraps llama.cpp which is single-threaded.
/// All access must be serialized via @SLlamaActor isolation.
public final class SLlamaSampler: PLlamaSampler {
    // MARK: Properties

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    var sampler: SLlamaSamplerPointer?

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    let context: SLlamaContext

    // MARK: Computed Properties

    /// Get the underlying C sampler pointer for direct API access
    public var cSampler: SLlamaSamplerPointer? {
        sampler
    }

    /// Get the sampler name
    /// - Returns: The name of the sampler, or nil if not available
    public var name: String? {
        guard let sampler else { return nil }
        return String(cString: llama_sampler_name(sampler))
    }

    // MARK: Lifecycle

    /// Initialize with a context
    /// - Parameter context: The llama context to use for sampling
    public init(context: SLlamaContext) {
        self.context = context
    }

    deinit {
        if let sampler {
            llama_sampler_free(sampler)
        }
    }

    // MARK: Core Sampling Functions

    /// Accept a token (updates internal state of certain samplers)
    /// - Parameter token: The token to accept
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func accept(_ token: SLlamaToken) {
        guard let sampler else { return }
        llama_sampler_accept(sampler, token)
    }

    /// Apply the sampler to token data array
    /// - Parameter tokenDataArray: The token data array to apply sampling to
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func apply(to tokenDataArray: SLlamaTokenDataArrayPointer) {
        guard let sampler else { return }
        llama_sampler_apply(sampler, tokenDataArray)
    }

    /// Reset the sampler state
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func reset() {
        guard let sampler else { return }
        llama_sampler_reset(sampler)
    }

    /// Clone the sampler
    /// - Returns: A new sampler instance, or nil if cloning failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func clone() -> PLlamaSampler? {
        guard let sampler else { return nil }
        guard let clonedSampler = llama_sampler_clone(sampler) else { return nil }

        let newSampler = SLlamaSampler(context: context)
        newSampler.sampler = clonedSampler
        return newSampler
    }

    /// Sample a token from the context
    ///
    /// **ARCHITECTURAL DECISION**: This method doesn't accept lastTokens parameter because
    /// modern llama.cpp uses stateful samplers for repetition control.
    ///
    /// **Why lastTokens was removed**:
    /// - Modern llama.cpp samplers maintain internal state via `llama_sampler_accept(token)`
    /// - Repetition penalties are handled by dedicated penalty samplers, not manual token tracking
    /// - Stateful approach provides better performance and accuracy than manual implementations
    /// - Reduces API confusion by having a single clear responsibility per function
    ///
    /// **For repetition control, use**:
    /// ```swift
    /// // ✅ Modern approach - stateful penalty sampler
    /// let penaltySampler = SLlamaSampler.penalties(
    ///     context: context,
    ///     penaltyLastN: 64,      // Automatically tracks last 64 tokens
    ///     penaltyRepeat: 1.1,    // Repetition penalty
    ///     penaltyFreq: 0.0,      // Frequency penalty
    ///     penaltyPresent: 0.0    // Presence penalty
    /// )
    /// let token = penaltySampler.sampleFromIndex(-1)
    /// penaltySampler.accept(token)  // Updates internal state
    ///
    /// // ❌ Or use the manual method if you need custom logic
    /// sampler.sampleWithRepetitionPenalty(1.1, lastTokens: previousTokens)
    /// ```
    ///
    /// - Returns: The sampled token ID, or nil if sampling failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func sample() -> SLlamaToken? {
        // Use the proper llama_sampler_sample API which handles all the internal state correctly
        // This samples from the last token position (-1)
        sampleFromIndex(-1)
    }

    /// Sample a token from the provided token data array (PLlamaSampler protocol method)
    /// - Parameter tokenDataArray: Pre-built token data array to sample from
    /// - Returns: The sampled token ID
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func sample(_ tokenDataArray: SLlamaTokenDataArrayPointer) -> SLlamaToken {
        guard let sampler else {
            // Fallback to greedy selection if no sampler configured
            return tokenDataArray.pointee.data[Int(tokenDataArray.pointee.selected)].id
        }

        // Apply the sampler to modify probabilities
        llama_sampler_apply(sampler, tokenDataArray)

        // Return the selected token (sampler sets the selected index)
        return tokenDataArray.pointee.data[Int(tokenDataArray.pointee.selected)].id
    }

    // MARK: - Convenience Sampling Methods

    /// Sample with temperature scaling
    ///
    /// **ARCHITECTURAL NOTE**: Temperature scaling is applied to logits before sampling,
    /// making them more (higher temp) or less (lower temp) random. This is a preprocessing
    /// step that doesn't require token history, unlike repetition penalties.
    ///
    /// **Why no lastTokens parameter**:
    /// - Temperature only affects the current sampling decision, not historical context
    /// - For repetition control, use dedicated penalty samplers (see sample() method docs)
    /// - Keeps the API focused on single responsibility (temperature scaling only)
    ///
    /// - Parameter temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
    /// - Returns: The sampled token ID, or nil if sampling failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func sampleWithTemperature(_ temperature: Float) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }

        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }

        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0

        if vocabSize == 0 { return nil }

        // Create token data array with temperature scaling
        var candidates = [SLlamaTokenData]()
        for i in 0 ..< vocabSize {
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
    ///
    /// **ARCHITECTURAL NOTE**: Top-k filtering selects only the k most likely tokens
    /// before sampling. This is a vocabulary filtering operation that doesn't depend
    /// on token history, unlike repetition penalties.
    ///
    /// **Why no lastTokens parameter**:
    /// - Top-k only considers current logits, not historical token patterns
    /// - For repetition control, use penalty samplers in a sampler chain
    /// - Maintains single responsibility principle (vocabulary filtering only)
    ///
    /// - Parameter k: Number of top tokens to consider
    /// - Returns: The sampled token ID, or nil if sampling failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func sampleTopK(_ k: Int) -> SLlamaToken? {
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
    ///
    /// **ARCHITECTURAL NOTE**: Top-p filtering selects tokens whose cumulative probability
    /// reaches the threshold p. This is a dynamic vocabulary filtering based on probability
    /// distribution, not token history.
    ///
    /// **Why no lastTokens parameter**:
    /// - Top-p only considers current probability distribution, not historical patterns
    /// - For repetition control, use penalty samplers in a sampler chain
    /// - Keeps the method focused on nucleus sampling logic only
    ///
    /// - Parameter p: Cumulative probability threshold (0.0 to 1.0)
    /// - Returns: The sampled token ID, or nil if sampling failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func sampleTopP(_ p: Float) -> SLlamaToken? {
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

        // Sort by logit value
        candidates.sort { $0.logit > $1.logit }

        // Calculate cumulative probabilities
        var cumulativeProb = 0.0
        var selectedCandidates: [SLlamaTokenData] = []

        for candidate in candidates {
            let prob = exp(Double(candidate.logit))
            cumulativeProb += prob
            selectedCandidates.append(candidate)

            if cumulativeProb >= Double(p) {
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

    /// Sample with repetition penalty (manual implementation)
    ///
    /// **ARCHITECTURAL NOTE**: This method provides a manual implementation of repetition
    /// penalty for cases where you need custom logic or can't use stateful samplers.
    ///
    /// **When to use this vs. penalty samplers**:
    /// ✅ Use this when:
    /// - You need custom penalty logic beyond what penalty samplers provide
    /// - You're implementing experimental penalty algorithms
    /// - You need precise control over which tokens get penalized
    ///
    /// ✅ Use penalty samplers when:
    /// - You want standard repetition/frequency/presence penalties
    /// - You want better performance (C implementation vs Swift loops)
    /// - You want automatic token history management
    /// - You're building sampler chains
    ///
    /// **Implementation details**:
    /// - Manually multiplies logits by penalty factor for tokens in lastTokens
    /// - penalty > 1.0 reduces probability of repeated tokens
    /// - penalty < 1.0 increases probability of repeated tokens (unusual)
    ///
    /// - Parameters:
    ///   - penalty: Repetition penalty factor (1.0 = no penalty, >1.0 = penalty)
    ///   - lastTokens: Array of last tokens to penalize
    /// - Returns: The sampled token ID, or nil if sampling failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
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
        for i in 0 ..< vocabSize {
            var logit = logits[Int(i)]

            // Apply repetition penalty to tokens that appear in lastTokens
            // **PERFORMANCE NOTE**: This is O(n*m) where n=vocab_size, m=lastTokens.count
            // For better performance with large token histories, use penalty samplers
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
