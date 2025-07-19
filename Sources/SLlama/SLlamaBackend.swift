import Foundation
import llama

// MARK: - SLlamaBackend

/// A wrapper for llama backend management and performance optimization
public class SLlamaBackend {
    // MARK: Static Properties

    @MainActor private static var _isInitialized = false

    // MARK: Static Computed Properties

    /// Check if the backend is initialized
    /// - Returns: true if the backend is initialized, false otherwise
    @MainActor
    public static var isInitialized: Bool {
        _isInitialized
    }

    // MARK: Static Functions

    /// Initialize the llama backend
    /// This should be called before any other llama operations
    @MainActor
    public static func initialize() {
        if !_isInitialized {
            llama_backend_init()
            _isInitialized = true
        }
    }

    /// Free the llama backend
    /// This should be called when the application is shutting down
    @MainActor
    public static func free() {
        if _isInitialized {
            llama_backend_free()
            _isInitialized = false
        }
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
}
