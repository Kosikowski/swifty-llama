import Foundation

// MARK: - Performance Utilities

/// Performance monitoring and benchmarking utilities
public class LlamaPerformance {
    
    private let context: LlamaContext?
    
    public init(context: LlamaContext? = nil) {
        self.context = context
    }
    
    // MARK: - Benchmarking
    
    /// Benchmark model loading performance
    /// - Parameters:
    ///   - modelPath: Path to the model file
    ///   - iterations: Number of iterations to run
    /// - Returns: Loading benchmark results, or nil if benchmarking failed
    public func benchmarkModelLoading(
        modelPath: String,
        iterations: Int = 5
    ) -> LoadingBenchmarkResults? {
        var totalLoadTime: TimeInterval = 0
        var loadTimes: [TimeInterval] = []
        
        for _ in 0..<iterations {
            let startTime = CFAbsoluteTimeGetCurrent()
            
            guard let model = LlamaModel(modelPath: modelPath) else { continue }
            guard let _ = LlamaContext(model: model) else { continue }
            
            let endTime = CFAbsoluteTimeGetCurrent()
            let loadTime = endTime - startTime
            
            totalLoadTime += loadTime
            loadTimes.append(loadTime)
        }
        
        let avgLoadTime = totalLoadTime / Double(iterations)
        let minLoadTime = loadTimes.min() ?? 0
        let maxLoadTime = loadTimes.max() ?? 0
        
        return LoadingBenchmarkResults(
            averageLoadTime: avgLoadTime,
            minimumLoadTime: minLoadTime,
            maximumLoadTime: maxLoadTime,
            iterations: iterations,
            totalLoadTime: totalLoadTime
        )
    }
    
    // MARK: - Profiling
    
    /// Profile memory usage during a custom operation
    /// - Parameters:
    ///   - operation: The operation to profile
    ///   - maxTokens: Maximum tokens to generate (for reference)
    /// - Returns: Memory profile results, or nil if profiling failed
    public func profileMemoryUsage(
        operation: () -> Void,
        maxTokens: Int = 100
    ) -> MemoryProfileResults? {
        let initialMemory = getCurrentMemoryUsage()
        
        // Run the operation
        operation()
        
        let finalMemory = getCurrentMemoryUsage()
        
        return MemoryProfileResults(
            initialMemory: initialMemory,
            finalMemory: finalMemory,
            memoryIncrease: finalMemory - initialMemory,
            tokenCount: maxTokens,
            memorySnapshots: []
        )
    }
    
    /// Profile CPU usage during a custom operation
    /// - Parameters:
    ///   - operation: The operation to profile
    ///   - maxTokens: Maximum tokens to generate (for reference)
    /// - Returns: CPU profile results, or nil if profiling failed
    public func profileCPUUsage(
        operation: () -> Void,
        maxTokens: Int = 100
    ) -> CPUProfileResults? {
        let startTime = CFAbsoluteTimeGetCurrent()
        let startCPU = getCurrentCPUUsage()
        
        // Run the operation
        operation()
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let endCPU = getCurrentCPUUsage()
        
        return CPUProfileResults(
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
    public func startMonitoring() -> PerformanceMonitor? {
        return PerformanceMonitor(context: context)
    }
    
    /// Get current memory usage
    /// - Returns: Current memory usage in bytes
    private func getCurrentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }
    
    /// Get current CPU usage
    /// - Returns: Current CPU usage as a percentage
    private func getCurrentCPUUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return kerr == KERN_SUCCESS ? Double(info.user_time.seconds) : 0.0
    }
}

// MARK: - Performance Monitor

/// Real-time performance monitoring
public class PerformanceMonitor {
    
    private let context: LlamaContext?
    private var isMonitoring = false
    private var monitoringTimer: Timer?
    private var metrics: [PerformanceMetric] = []
    
    public init(context: LlamaContext?) {
        self.context = context
    }
    
    /// Start monitoring
    public func start() {
        guard !isMonitoring else { return }
        
        isMonitoring = true
        monitoringTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.recordMetrics()
        }
    }
    
    /// Stop monitoring
    public func stop() {
        isMonitoring = false
        monitoringTimer?.invalidate()
        monitoringTimer = nil
    }
    
    /// Get monitoring results
    /// - Returns: Array of recorded performance metrics
    public func getResults() -> [PerformanceMetric] {
        return metrics
    }
    
    /// Clear monitoring results
    public func clearResults() {
        metrics.removeAll()
    }
    
    /// Record current metrics
    private func recordMetrics() {
        let metric = PerformanceMetric(
            timestamp: Date(),
            memoryUsage: getCurrentMemoryUsage(),
            cpuUsage: getCurrentCPUUsage(),
            activeThreads: getActiveThreadCount()
        )
        metrics.append(metric)
    }
    
    /// Get current memory usage
    private func getCurrentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }
    
    /// Get current CPU usage
    private func getCurrentCPUUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return kerr == KERN_SUCCESS ? Double(info.user_time.seconds) : 0.0
    }
    
    /// Get active thread count
    private func getActiveThreadCount() -> Int {
        var threadList: thread_act_array_t?
        var threadCount: mach_msg_type_number_t = 0
        
        let result = task_threads(mach_task_self_, &threadList, &threadCount)
        
        if result == KERN_SUCCESS {
            vm_deallocate(mach_task_self_, vm_address_t(UInt(bitPattern: threadList)), vm_size_t(threadCount) * vm_size_t(MemoryLayout<thread_t>.size))
            return Int(threadCount)
        }
        
        return 0
    }
}

// MARK: - Result Types

/// Benchmark results for model loading performance
public struct LoadingBenchmarkResults {
    public let averageLoadTime: TimeInterval
    public let minimumLoadTime: TimeInterval
    public let maximumLoadTime: TimeInterval
    public let iterations: Int
    public let totalLoadTime: TimeInterval
}

/// Memory profile results
public struct MemoryProfileResults {
    public let initialMemory: UInt64
    public let finalMemory: UInt64
    public let memoryIncrease: UInt64
    public let tokenCount: Int
    public let memorySnapshots: [MemorySnapshot]
}

/// CPU profile results
public struct CPUProfileResults {
    public let startCPU: Double
    public let endCPU: Double
    public let averageCPU: Double
    public let duration: TimeInterval
    public let tokenCount: Int
    public let cpuSnapshots: [CPUSnapshot]
}

/// Memory snapshot
public struct MemorySnapshot {
    public let tokenIndex: Int
    public let memoryUsage: UInt64
    public let timestamp: TimeInterval
}

/// CPU snapshot
public struct CPUSnapshot {
    public let tokenIndex: Int
    public let cpuUsage: Double
    public let timestamp: TimeInterval
}

/// Performance metric
public struct PerformanceMetric {
    public let timestamp: Date
    public let memoryUsage: UInt64
    public let cpuUsage: Double
    public let activeThreads: Int
}

// MARK: - Extension to LlamaContext

public extension LlamaContext {
    
    /// Get performance utilities interface
    /// - Returns: LlamaPerformance instance for this context
    func performance() -> LlamaPerformance {
        return LlamaPerformance(context: self)
    }
    
    /// Start performance monitoring
    /// - Returns: Performance monitor instance, or nil if monitoring failed
    func startPerformanceMonitoring() -> PerformanceMonitor? {
        return performance().startMonitoring()
    }
} 