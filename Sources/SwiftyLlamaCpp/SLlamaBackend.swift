import Foundation
import llama

/// A wrapper for llama backend management and performance optimization
public class SLlamaBackend {
    
    /// Initialize the llama backend
    /// This should be called before any other llama operations
    public static func initialize() {
        llama_backend_init()
    }
    
    /// Free the llama backend
    /// This should be called when the application is shutting down
    public static func free() {
        llama_backend_free()
    }
    
    /// Check if the backend is initialized
    /// - Returns: true if the backend is initialized, false otherwise
    public static var isInitialized: Bool {
        // Note: This is a placeholder implementation
        // In a real implementation, you might track initialization state
        return true
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
        return Int32(ProcessInfo.processInfo.activeProcessorCount)
    }
    
    /// Configure context for optimal performance
    /// - Parameters:
    ///   - useOptimalThreads: Whether to use optimal thread count
    ///   - customThreads: Custom thread count (ignored if useOptimalThreads is true)
    func optimizeForPerformance(useOptimalThreads: Bool = true, customThreads: Int32? = nil) {
        let threadCount: Int32
        
        if useOptimalThreads {
            threadCount = Self.optimalThreadCount()
        } else {
            threadCount = customThreads ?? 4
        }
        
        setThreadCount(threadCount)
    }
} 