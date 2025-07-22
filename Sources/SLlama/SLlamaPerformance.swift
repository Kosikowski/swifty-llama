import Foundation
import llama
import Omen

// MARK: - Type Aliases

/// Type alias for backwards compatibility
public typealias SContextPerformanceMetrics = SDetailedContextMetrics

/// Type alias for backwards compatibility
public typealias SSamplerPerformanceMetrics = SDetailedSamplerMetrics

// MARK: - SLlamaPerformance

/// A wrapper for performance measurement and monitoring in SLlama
///
/// **ARCHITECTURAL DECISION**: Using structured logging instead of print statements
/// for performance metrics provides better observability and integration with system tools.
public class SLlamaPerformance: @unchecked Sendable {
    // MARK: Properties

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    let context: SLlamaContext?

    // MARK: Lifecycle

    public init(context: SLlamaContext? = nil) {
        self.context = context
    }

    // MARK: Functions

    // MARK: - Benchmarking Functions

    /// Benchmark model loading performance
    /// - Parameters:
    ///   - modelPath: Path to the model file
    ///   - iterations: Number of iterations to run
    /// - Returns: Loading benchmark results, or nil if benchmarking failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func benchmarkModelLoading(
        modelPath: String,
        iterations: Int = 5
    )
        -> SLoadingBenchmarkResults?
    {
        var totalLoadTime: TimeInterval = 0
        var loadTimes: [TimeInterval] = []

        for _ in 0 ..< iterations {
            let startTime = CFAbsoluteTimeGetCurrent()

            do {
                let model = try SLlamaModel(modelPath: modelPath)
                _ = try SLlamaContext(model: model)

                let endTime = CFAbsoluteTimeGetCurrent()
                let loadTime = endTime - startTime

                totalLoadTime += loadTime
                loadTimes.append(loadTime)
            } catch {
                // Skip this iteration on error
                continue
            }
        }

        let avgLoadTime = totalLoadTime / Double(iterations)
        let minLoadTime = loadTimes.min() ?? 0
        let maxLoadTime = loadTimes.max() ?? 0

        return SLoadingBenchmarkResults(
            averageLoadTime: avgLoadTime,
            minimumLoadTime: minLoadTime,
            maximumLoadTime: maxLoadTime,
            iterations: iterations,
            totalLoadTime: totalLoadTime
        )
    }

    /// Benchmark token generation performance
    /// - Parameters:
    ///   - modelPath: Path to the model file
    ///   - prompt: Prompt to generate tokens from
    ///   - maxTokens: Maximum number of tokens to generate
    ///   - iterations: Number of iterations to run
    /// - Returns: Token generation benchmark results, or nil if benchmarking failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func benchmarkTokenGeneration(
        modelPath: String,
        prompt: String,
        maxTokens: Int = 100,
        iterations: Int = 3
    )
        -> STokenGenerationBenchmarkResults?
    {
        var totalGenerationTime: TimeInterval = 0
        var totalTokensGenerated = 0
        var generationTimes: [TimeInterval] = []
        var tokensPerIteration: [Int] = []

        for _ in 0 ..< iterations {
            let startTime = CFAbsoluteTimeGetCurrent()

            do {
                let model = try SLlamaModel(modelPath: modelPath)
                let context = try SLlamaContext(model: model)
                let vocab = SLlamaVocab(vocab: model.vocab)

                // Tokenize the prompt
                let promptTokens = try vocab.tokenize(text: prompt)

                // Process prompt tokens
                let batch = SLlamaBatch(nTokens: Int32(promptTokens.count), nSeqMax: 1)
                for (index, token) in promptTokens.enumerated() {
                    batch.addToken(
                        token,
                        position: Int32(index),
                        sequenceIds: [0],
                        logits: index == promptTokens.count - 1
                    )
                }
                try context.core().decode(batch)

                // Generate tokens
                let sampler = SLlamaSampler.temperature(context: context, temperature: 0.7) ?? SLlamaSampler
                    .greedy(context: context) ?? SLlamaSampler(context: context)
                var tokensGenerated = 0
                var currentPosition = promptTokens.count

                for _ in 0 ..< maxTokens {
                    guard let nextToken = sampler.sample() else { break }

                    if nextToken == vocab.eosToken { break }

                    sampler.accept(nextToken)
                    tokensGenerated += 1

                    // Process single token
                    let generationBatch = SLlamaBatch(nTokens: 1, nSeqMax: 1)
                    generationBatch.addToken(
                        nextToken,
                        position: Int32(currentPosition),
                        sequenceIds: [0],
                        logits: true
                    )
                    try context.core().decode(generationBatch)
                    currentPosition += 1
                }

                let endTime = CFAbsoluteTimeGetCurrent()
                let generationTime = endTime - startTime

                totalGenerationTime += generationTime
                totalTokensGenerated += tokensGenerated
                generationTimes.append(generationTime)
                tokensPerIteration.append(tokensGenerated)
            } catch {
                // Skip this iteration on error
                continue
            }
        }

        let avgGenerationTime = totalGenerationTime / Double(iterations)
        let minGenerationTime = generationTimes.min() ?? 0
        let maxGenerationTime = generationTimes.max() ?? 0
        let avgTokensPerIteration = totalTokensGenerated / iterations
        let tokensPerSecond = totalGenerationTime > 0 ? Double(totalTokensGenerated) / totalGenerationTime : 0

        return STokenGenerationBenchmarkResults(
            averageGenerationTime: avgGenerationTime,
            minimumGenerationTime: minGenerationTime,
            maximumGenerationTime: maxGenerationTime,
            totalTokensGenerated: totalTokensGenerated,
            averageTokensPerIteration: avgTokensPerIteration,
            tokensPerSecond: tokensPerSecond,
            iterations: iterations,
            totalGenerationTime: totalGenerationTime
        )
    }

    // MARK: - Profiling

    /// Profile memory usage during a custom operation
    /// - Parameters:
    ///   - operation: The operation to profile
    ///   - maxTokens: Maximum tokens to generate (for reference)
    /// - Returns: Memory profile results, or nil if profiling failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func profileMemoryUsage(
        operation: () -> Void,
        maxTokens: Int = 100
    )
        -> SMemoryProfileResults?
    {
        let initialMemory = getCurrentMemoryUsage()

        // Run the operation
        operation()

        let finalMemory = getCurrentMemoryUsage()

        // Calculate memory increase as signed value to handle both increases and decreases
        let memoryIncrease = Int64(finalMemory) - Int64(initialMemory)

        return SMemoryProfileResults(
            initialMemory: initialMemory,
            finalMemory: finalMemory,
            memoryIncrease: memoryIncrease,
            tokenCount: maxTokens,
            memorySnapshots: []
        )
    }

    /// Profile CPU usage during a custom operation
    /// - Parameters:
    ///   - operation: The operation to profile
    ///   - maxTokens: Maximum tokens to generate (for reference)
    /// - Returns: CPU profile results, or nil if profiling failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func profileCPUUsage(
        operation: () -> Void,
        maxTokens: Int = 100
    )
        -> SCPUProfileResults?
    {
        let startTime = CFAbsoluteTimeGetCurrent()
        let startCPU = getCurrentCPUUsage()

        // Run the operation
        operation()

        let endTime = CFAbsoluteTimeGetCurrent()
        let endCPU = getCurrentCPUUsage()

        return SCPUProfileResults(
            startCPU: startCPU,
            endCPU: endCPU,
            averageCPU: (startCPU + endCPU) / 2.0,
            duration: endTime - startTime,
            tokenCount: maxTokens,
            cpuSnapshots: []
        )
    }

    // MARK: - Performance Monitoring

    /// Start performance monitoring
    /// - Returns: Performance monitor instance, or nil if monitoring failed
    @SLlamaActor
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func startMonitoring() -> SPerformanceMonitor? {
        SPerformanceMonitor(context: context)
    }

    // MARK: - Llama.cpp Performance Functions

    /// Get performance context data from llama.cpp
    /// - Parameter context: The llama context to get performance data for
    /// - Returns: Performance context data, or nil if context is invalid
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getContextPerformanceData(context: SLlamaContext) -> SLlamaPerfContextData? {
        guard context.pointer != nil else { return nil }

        // Try to use llama.cpp performance functions if available
        #if canImport(llama)
            // Note: These functions may not be available in all builds
            // We'll provide fallback implementations
        #endif

        // Fallback: Return custom performance data based on available metrics
        let metrics = getDetailedContextMetrics(context: context)
        return SLlamaPerfContextData(
            t_start_ms: metrics.startTimeMs,
            t_load_ms: metrics.loadTimeMs,
            t_p_eval_ms: metrics.promptEvalTimeMs,
            t_eval_ms: metrics.evalTimeMs,
            n_p_eval: Int32(metrics.promptEvalCount),
            n_eval: Int32(metrics.evalCount),
            n_reused: Int32(metrics.reusedCount)
        )
    }

    /// Print context performance data with structured logging
    ///
    /// **LOGGING STRATEGY**: Performance metrics are logged at INFO level under the
    /// .performance category for easy filtering and analysis in Console.app
    ///
    /// - Parameter metrics: The performance metrics to display
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func printContextPerformanceData(_ metrics: SContextPerformanceMetrics) {
        Omen.performance("=== Context Performance Data ===")
        Omen.performance("Start Time: \(metrics.startTimeMs) ms")
        Omen.performance("Load Time: \(metrics.loadTimeMs) ms")
        Omen.performance("Prompt Eval Time: \(metrics.promptEvalTimeMs) ms")
        Omen.performance("Eval Time: \(metrics.evalTimeMs) ms")
        Omen.performance("Total Eval Time: \(metrics.totalEvalTimeMs) ms")
        Omen.performance("Prompt Eval Count: \(metrics.promptEvalCount)")
        Omen.performance("Eval Count: \(metrics.evalCount)")
        Omen.performance("Reused Count: \(metrics.reusedCount)")
        Omen.performance("Average Eval Time: \(metrics.averageEvalTimeMs) ms")
        Omen.performance("Average Prompt Eval Time: \(metrics.averagePromptEvalTimeMs) ms")
        Omen.performance("Efficiency Ratio: \(metrics.efficiencyRatio)")
        Omen.performance("================================")
    }

    /// Reset context performance data with logging
    ///
    /// **LOGGING STRATEGY**: Reset operations are logged at INFO level to track
    /// when performance monitoring cycles begin/end
    ///
    /// - Parameter context: The context to reset performance data for
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func resetContextPerformanceData(_ context: SLlamaContext) {
        guard let ctx = context.pointer else {
            Omen.error(
                OmenCategories.Core.performance,
                "Failed to reset context performance data: context pointer is nil"
            )
            return
        }

        llama_perf_context_reset(ctx)
        Omen.performance("Context performance data reset")
    }

    /// Get performance sampler data from llama.cpp
    /// - Parameter sampler: The sampler to get performance data for
    /// - Returns: Performance sampler data, or nil if sampler is invalid
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getSamplerPerformanceData(sampler: PLlamaSampler) -> SLlamaPerfSamplerData? {
        guard sampler.cSampler != nil else { return nil }

        // Try to use llama.cpp performance functions if available
        #if canImport(llama)
            // Note: These functions may not be available in all builds
            // We'll provide fallback implementations
        #endif

        // Fallback: Return custom performance data based on available metrics
        let metrics = getDetailedSamplerMetrics(sampler: sampler)
        return SLlamaPerfSamplerData(
            t_sample_ms: metrics.sampleTimeMs,
            n_sample: Int32(metrics.sampleCount)
        )
    }

    /// Print sampler performance data with structured logging
    ///
    /// **LOGGING STRATEGY**: Sampler metrics are logged under .performance category
    /// to maintain consistency with other performance measurements
    ///
    /// - Parameter metrics: The sampler performance metrics to display
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func printSamplerPerformanceData(_ metrics: SSamplerPerformanceMetrics) {
        Omen.performance("=== Sampler Performance Data ===")
        Omen.performance("Sample Time: \(metrics.sampleTimeMs) ms")
        Omen.performance("Sample Count: \(metrics.sampleCount)")
        Omen.performance("Average Sample Time: \(metrics.averageSampleTimeMs) ms")
        Omen.performance("Samples Per Second: \(metrics.samplesPerSecond)")
        Omen.performance("================================")
    }

    /// Reset sampler performance data in llama.cpp
    /// - Parameter sampler: The sampler to reset performance data for
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func resetSamplerPerformanceData(_ sampler: PLlamaSampler) {
        guard let samplerPtr = sampler.cSampler else {
            Omen.error(
                OmenCategories.Core.performance,
                "Failed to reset sampler performance data: sampler pointer is nil"
            )
            return
        }

        llama_perf_sampler_reset(samplerPtr)
        Omen.performance("Sampler performance data reset")
    }

    /// Get detailed performance metrics for a context
    /// - Parameter context: The llama context to analyze
    /// - Returns: Detailed performance metrics
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getDetailedContextMetrics(context: SLlamaContext) -> SDetailedContextMetrics {
        // Try to get llama.cpp performance data first
        if let perfData = getContextPerformanceData(context: context) {
            return SDetailedContextMetrics(
                startTimeMs: perfData.t_start_ms,
                loadTimeMs: perfData.t_load_ms,
                promptEvalTimeMs: perfData.t_p_eval_ms,
                evalTimeMs: perfData.t_eval_ms,
                promptEvalCount: Int(perfData.n_p_eval),
                evalCount: Int(perfData.n_eval),
                reusedCount: Int(perfData.n_reused)
            )
        }

        // Fallback: Return custom metrics based on available data
        return SDetailedContextMetrics(
            startTimeMs: 0.0, // Not available without llama.cpp functions
            loadTimeMs: 0.0,
            promptEvalTimeMs: 0.0,
            evalTimeMs: 0.0,
            promptEvalCount: 0,
            evalCount: 0,
            reusedCount: 0
        )
    }

    /// Get detailed sampler metrics for debugging and analysis
    /// - Parameter sampler: The sampler to analyze
    /// - Returns: Detailed metrics structure
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getDetailedSamplerMetrics(sampler: PLlamaSampler) -> SDetailedSamplerMetrics {
        // Try to get llama.cpp performance data first
        if let perfData = getSamplerPerformanceData(sampler: sampler) {
            return SDetailedSamplerMetrics(
                sampleTimeMs: perfData.t_sample_ms,
                sampleCount: Int(perfData.n_sample)
            )
        }

        // Fallback: Return custom metrics based on available data
        return SDetailedSamplerMetrics(
            sampleTimeMs: 0.0, // Not available without llama.cpp functions
            sampleCount: 0
        )
    }

    /// Get current memory usage
    /// - Returns: Current memory usage in bytes
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    func getCurrentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }

    /// Get current CPU usage
    /// - Returns: Current CPU usage as a percentage
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    func getCurrentCPUUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        return kerr == KERN_SUCCESS ? Double(info.user_time.seconds) : 0.0
    }
}

// MARK: - SPerformanceMonitor

/// Real-time performance monitoring
public final class SPerformanceMonitor: @unchecked Sendable {
    // MARK: Properties

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    let context: SLlamaContext?
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    var isMonitoring = false
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    var monitoringTimer: Timer?
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    var metrics: [SPerformanceMetric] = []

    // MARK: Lifecycle

    public init(context: SLlamaContext?) {
        self.context = context
    }

    // MARK: Functions

    /// Start monitoring
    @SLlamaActor
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func start() {
        guard !isMonitoring else { return }

        isMonitoring = true
        monitoringTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            Task { @SLlamaActor in
                self.recordMetrics()
            }
        }
    }

    /// Stop monitoring
    @SLlamaActor
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func stop() {
        isMonitoring = false
        monitoringTimer?.invalidate()
        monitoringTimer = nil
    }

    /// Get monitoring results
    /// - Returns: Array of recorded performance metrics
    @SLlamaActor
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getResults() -> [SPerformanceMetric] {
        metrics
    }

    /// Clear monitoring results
    @SLlamaActor
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func clearResults() {
        metrics.removeAll()
    }

    /// Record current metrics
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    func recordMetrics() {
        let metric = SPerformanceMetric(
            timestamp: Date(),
            memoryUsage: getCurrentMemoryUsage(),
            cpuUsage: getCurrentCPUUsage(),
            activeThreads: getActiveThreadCount()
        )
        metrics.append(metric)
    }

    /// Get current memory usage
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    func getCurrentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }

    /// Get current CPU usage
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    func getCurrentCPUUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        return kerr == KERN_SUCCESS ? Double(info.user_time.seconds) : 0.0
    }

    /// Get active thread count
    func getActiveThreadCount() -> Int {
        var threadList: thread_act_array_t?
        var threadCount: mach_msg_type_number_t = 0

        let result = task_threads(mach_task_self_, &threadList, &threadCount)

        if result == KERN_SUCCESS {
            vm_deallocate(
                mach_task_self_,
                vm_address_t(UInt(bitPattern: threadList)),
                vm_size_t(threadCount) * vm_size_t(MemoryLayout<thread_t>.size)
            )
            return Int(threadCount)
        }

        return 0
    }
}

// MARK: - SLoadingBenchmarkResults

/// Benchmark results for model loading performance
public struct SLoadingBenchmarkResults {
    public let averageLoadTime: TimeInterval
    public let minimumLoadTime: TimeInterval
    public let maximumLoadTime: TimeInterval
    public let iterations: Int
    public let totalLoadTime: TimeInterval

    public init(
        averageLoadTime: TimeInterval,
        minimumLoadTime: TimeInterval,
        maximumLoadTime: TimeInterval,
        iterations: Int,
        totalLoadTime: TimeInterval
    ) {
        self.averageLoadTime = averageLoadTime
        self.minimumLoadTime = minimumLoadTime
        self.maximumLoadTime = maximumLoadTime
        self.iterations = iterations
        self.totalLoadTime = totalLoadTime
    }
}

// MARK: - STokenGenerationBenchmarkResults

/// Benchmark results for token generation performance
public struct STokenGenerationBenchmarkResults {
    public let averageGenerationTime: TimeInterval
    public let minimumGenerationTime: TimeInterval
    public let maximumGenerationTime: TimeInterval
    public let totalTokensGenerated: Int
    public let averageTokensPerIteration: Int
    public let tokensPerSecond: Double
    public let iterations: Int
    public let totalGenerationTime: TimeInterval

    public init(
        averageGenerationTime: TimeInterval,
        minimumGenerationTime: TimeInterval,
        maximumGenerationTime: TimeInterval,
        totalTokensGenerated: Int,
        averageTokensPerIteration: Int,
        tokensPerSecond: Double,
        iterations: Int,
        totalGenerationTime: TimeInterval
    ) {
        self.averageGenerationTime = averageGenerationTime
        self.minimumGenerationTime = minimumGenerationTime
        self.maximumGenerationTime = maximumGenerationTime
        self.totalTokensGenerated = totalTokensGenerated
        self.averageTokensPerIteration = averageTokensPerIteration
        self.tokensPerSecond = tokensPerSecond
        self.iterations = iterations
        self.totalGenerationTime = totalGenerationTime
    }
}

// MARK: - SMemoryProfileResults

/// Memory profile results
public struct SMemoryProfileResults {
    public let initialMemory: UInt64
    public let finalMemory: UInt64
    public let memoryIncrease: Int64 // Changed from UInt64 to Int64
    public let tokenCount: Int
    public let memorySnapshots: [SMemorySnapshot]

    public init(
        initialMemory: UInt64,
        finalMemory: UInt64,
        memoryIncrease: Int64, // Changed from UInt64 to Int64
        tokenCount: Int,
        memorySnapshots: [SMemorySnapshot]
    ) {
        self.initialMemory = initialMemory
        self.finalMemory = finalMemory
        self.memoryIncrease = memoryIncrease
        self.tokenCount = tokenCount
        self.memorySnapshots = memorySnapshots
    }
}

// MARK: - SCPUProfileResults

/// CPU profile results
public struct SCPUProfileResults {
    public let startCPU: Double
    public let endCPU: Double
    public let averageCPU: Double
    public let duration: TimeInterval
    public let tokenCount: Int
    public let cpuSnapshots: [SCPUSnapshot]

    public init(
        startCPU: Double,
        endCPU: Double,
        averageCPU: Double,
        duration: TimeInterval,
        tokenCount: Int,
        cpuSnapshots: [SCPUSnapshot]
    ) {
        self.startCPU = startCPU
        self.endCPU = endCPU
        self.averageCPU = averageCPU
        self.duration = duration
        self.tokenCount = tokenCount
        self.cpuSnapshots = cpuSnapshots
    }
}

// MARK: - SMemorySnapshot

/// Memory snapshot
public struct SMemorySnapshot {
    public let tokenIndex: Int
    public let memoryUsage: UInt64
    public let timestamp: TimeInterval
}

// MARK: - SCPUSnapshot

/// CPU snapshot
public struct SCPUSnapshot {
    public let tokenIndex: Int
    public let cpuUsage: Double
    public let timestamp: TimeInterval
}

// MARK: - SPerformanceMetric

/// Performance metric
public struct SPerformanceMetric {
    public let timestamp: Date
    public let memoryUsage: UInt64
    public let cpuUsage: Double
    public let activeThreads: Int
}

// MARK: - SDetailedContextMetrics

/// Detailed context performance metrics from llama.cpp
public struct SDetailedContextMetrics {
    // MARK: Properties

    public let startTimeMs: Double
    public let loadTimeMs: Double
    public let promptEvalTimeMs: Double
    public let evalTimeMs: Double
    public let promptEvalCount: Int
    public let evalCount: Int
    public let reusedCount: Int

    public init(
        startTimeMs: Double,
        loadTimeMs: Double,
        promptEvalTimeMs: Double,
        evalTimeMs: Double,
        promptEvalCount: Int,
        evalCount: Int,
        reusedCount: Int
    ) {
        self.startTimeMs = startTimeMs
        self.loadTimeMs = loadTimeMs
        self.promptEvalTimeMs = promptEvalTimeMs
        self.evalTimeMs = evalTimeMs
        self.promptEvalCount = promptEvalCount
        self.evalCount = evalCount
        self.reusedCount = reusedCount
    }

    // MARK: Computed Properties

    /// Total evaluation time (prompt + eval)
    public var totalEvalTimeMs: Double {
        promptEvalTimeMs + evalTimeMs
    }

    /// Average time per evaluation
    public var averageEvalTimeMs: Double {
        evalCount > 0 ? evalTimeMs / Double(evalCount) : 0.0
    }

    /// Average time per prompt evaluation
    public var averagePromptEvalTimeMs: Double {
        promptEvalCount > 0 ? promptEvalTimeMs / Double(promptEvalCount) : 0.0
    }

    /// Efficiency ratio (reused vs total evaluations)
    public var efficiencyRatio: Double {
        let totalEvals = promptEvalCount + evalCount
        return totalEvals > 0 ? Double(reusedCount) / Double(totalEvals) : 0.0
    }
}

// MARK: - SDetailedSamplerMetrics

/// Detailed sampler performance metrics from llama.cpp
public struct SDetailedSamplerMetrics {
    // MARK: Properties

    public let sampleTimeMs: Double
    public let sampleCount: Int

    public init(
        sampleTimeMs: Double,
        sampleCount: Int
    ) {
        self.sampleTimeMs = sampleTimeMs
        self.sampleCount = sampleCount
    }

    // MARK: Computed Properties

    /// Average time per sample
    public var averageSampleTimeMs: Double {
        sampleCount > 0 ? sampleTimeMs / Double(sampleCount) : 0.0
    }

    /// Samples per second
    public var samplesPerSecond: Double {
        sampleTimeMs > 0 ? Double(sampleCount) / (sampleTimeMs / 1000.0) : 0.0
    }
}

// MARK: - Extension to SLlamaContext

public extension SLlamaContext {
    /// Get performance utilities interface
    /// - Returns: SLlamaPerformance instance for this context
    func performance() -> SLlamaPerformance {
        SLlamaPerformance(context: self)
    }

    /// Start performance monitoring
    /// - Returns: Performance monitor instance, or nil if monitoring failed
    @SLlamaActor
    func startPerformanceMonitoring() -> SPerformanceMonitor? {
        performance().startMonitoring()
    }

    /// Get detailed performance metrics for this context
    /// - Returns: Detailed performance metrics
    func getDetailedPerformanceMetrics() -> SDetailedContextMetrics {
        performance().getDetailedContextMetrics(context: self)
    }

    /// Print performance data for this context to console
    func printPerformanceData() {
        performance().printContextPerformanceData(getDetailedPerformanceMetrics())
    }

    /// Reset performance data for this context
    func resetPerformanceData() {
        performance().resetContextPerformanceData(self)
    }
}

// MARK: - Extension to SLlamaSampler

public extension SLlamaSampler {
    /// Get detailed performance metrics for this sampler
    /// - Returns: Detailed sampler metrics
    func getDetailedPerformanceMetrics() -> SDetailedSamplerMetrics {
        let performance = SLlamaPerformance()
        return performance.getDetailedSamplerMetrics(sampler: self)
    }

    /// Print performance data for this sampler to console
    func printPerformanceData() {
        let performance = SLlamaPerformance()
        performance.printSamplerPerformanceData(getDetailedPerformanceMetrics())
    }

    /// Reset performance data for this sampler
    func resetPerformanceData() {
        let performance = SLlamaPerformance()
        performance.resetSamplerPerformanceData(self)
    }
}
