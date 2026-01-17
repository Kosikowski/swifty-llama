import Foundation
import llama

// MARK: - SLlamaContext Sampling Extension

public extension SLlamaContext {
    /// Create a sampler for this context
    /// - Returns: A SLlamaSampler instance
    func sampler() -> SLlamaSampler {
        SLlamaSampler(context: self)
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
        sampler().sampleWithTemperature(temperature)
    }

    /// Sample a token with top-k filtering
    /// - Parameter k: Number of top tokens to consider
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleTopK(_ k: Int) -> SLlamaToken? {
        sampler().sampleTopK(k)
    }

    /// Sample a token with top-p (nucleus) filtering
    /// - Parameter p: Cumulative probability threshold (0.0 to 1.0)
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleTopP(_ p: Float) -> SLlamaToken? {
        sampler().sampleTopP(p)
    }

    /// Sample a token with repetition penalty
    /// - Parameters:
    ///   - penalty: Repetition penalty factor (1.0 = no penalty, higher = more penalty)
    ///   - lastTokens: Array of last tokens to penalize
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithRepetitionPenalty(_ penalty: Float, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        sampler().sampleWithRepetitionPenalty(penalty, lastTokens: lastTokens)
    }
}
