import Foundation
import llama

// MARK: - SLlamaBackend

/// A wrapper for llama backend operations
public class SLlamaBackend: @unchecked Sendable {
    // MARK: Static Properties

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    @SLlamaActor static var _isInitialized = false

    // MARK: Static Computed Properties

    /// Check if the backend is initialized
    /// - Returns: true if the backend is initialized, false otherwise
    @SLlamaActor
    public static var isInitialized: Bool {
        _isInitialized
    }

    // MARK: Static Functions

    /// Initialize the llama backend
    /// This should be called before any other llama operations
    @SLlamaActor
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func initialize() {
        if !_isInitialized {
            llama_backend_init()
            _isInitialized = true
        }
    }

    /// Free the llama backend
    /// This should be called when the application is shutting down
    @SLlamaActor
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func free() {
        if _isInitialized {
            llama_backend_free()
            _isInitialized = false
        }
    }

    /// Initialize NUMA (Non-Uniform Memory Access) optimization
    /// Call this once after backend initialization for better performance on NUMA systems (multi-socket)
    /// - Parameter strategy: The NUMA strategy to use
    /// - Note: This should be called before creating any contexts for best results
    @SLlamaActor
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func initializeNUMA(_ strategy: SLlamaNUMAStrategy = .disabled) {
        llama_numa_init(strategy.rawStrategy)
    }

    /// Initialize NUMA with raw strategy value
    /// - Parameter strategy: The raw ggml_numa_strategy value
    @SLlamaActor
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func initializeNUMA(rawStrategy: SLlamaNUMAStrategyRaw) {
        llama_numa_init(rawStrategy)
    }
}

// MARK: - Extension to SLlamaContext for Performance Optimization

public extension SLlamaContext {
    /// Set the number of threads for inference
    /// - Parameters:
    ///   - threads: Number of threads for inference
    ///   - batchThreads: Number of threads for batch processing
    func setThreadCount(inference: Int32, batch: Int32) {
        guard let context = pointer else { return }
        llama_set_n_threads(context, inference, batch)
    }

    /// Set the number of threads for inference (same for both inference and batch)
    /// - Parameter threads: Number of threads
    func setThreadCount(_ threads: Int32) {
        setThreadCount(inference: threads, batch: threads)
    }

    /// Get optimal thread count for the current system
    /// - Returns: Recommended number of threads
    static func optimalThreadCount() -> Int32 {
        Int32(ProcessInfo.processInfo.activeProcessorCount)
    }

    /// Configure context for optimal performance
    /// - Parameters:
    ///   - useOptimalThreads: Whether to use optimal thread count
    ///   - customThreads: Custom thread count (ignored if useOptimalThreads is true)
    func optimizeForPerformance(useOptimalThreads: Bool = true, customThreads: Int32? = nil) {
        let threadCount: Int32 = if useOptimalThreads {
            Self.optimalThreadCount()
        } else {
            customThreads ?? 4
        }

        setThreadCount(threadCount)
    }

    /// Attach custom threadpools to this context
    /// - Parameters:
    ///   - threadpool: The main threadpool for inference operations
    ///   - batchThreadpool: The threadpool for batch processing (can be the same as threadpool)
    /// - Note: An automatic threadpool is created by ggml if not attached explicitly.
    ///         Use this for fine-grained control over thread management.
    func attachThreadpool(_ threadpool: SLlamaThreadpool, batchThreadpool: SLlamaThreadpool? = nil) {
        guard let context = pointer else { return }
        llama_attach_threadpool(context, threadpool, batchThreadpool ?? threadpool)
    }

    /// Detach any attached threadpools from this context
    /// - Note: After detaching, the context will use its default threading behavior
    func detachThreadpool() {
        guard let context = pointer else { return }
        llama_detach_threadpool(context)
    }
}
