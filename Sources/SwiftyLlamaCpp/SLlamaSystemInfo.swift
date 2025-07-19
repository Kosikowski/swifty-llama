import Foundation
import llama

/// A wrapper for llama.cpp system information and utility functions
public class SLlamaSystemInfo {
    
    /// Log detailed system information to stdout
    public static func logSystemInfo() {
        llama_print_system_info()
    }
    
    /// Get current time in microseconds
    /// - Returns: Current time in microseconds
    public static func getCurrentTimeMicroseconds() -> Int64 {
        return llama_time_us()
    }
    
    /// Get maximum number of available devices
    /// - Returns: Maximum number of devices
    public static func getMaxDevices() -> Int {
        return Int(llama_max_devices())
    }
    
    /// Get maximum number of parallel sequences
    /// - Returns: Maximum number of parallel sequences
    public static func getMaxParallelSequences() -> Int {
        return Int(llama_max_parallel_sequences())
    }
    
    /// Check if memory mapping is supported
    /// - Returns: True if mmap is supported
    public static func supportsMmap() -> Bool {
        return llama_supports_mmap()
    }
    
    /// Check if memory locking is supported
    /// - Returns: True if mlock is supported
    public static func supportsMlock() -> Bool {
        return llama_supports_mlock()
    }
    
    /// Check if GPU offloading is supported
    /// - Returns: True if GPU offloading is supported
    public static func supportsGpuOffload() -> Bool {
        return llama_supports_gpu_offload()
    }
    
    /// Check if RPC is supported
    /// - Returns: True if RPC is supported
    public static func supportsRpc() -> Bool {
        return llama_supports_rpc()
    }
}

// MARK: - System Information Struct

/// Comprehensive system information
public struct SLlamaSystemCapabilities {
    public let maxDevices: Int
    public let maxParallelSequences: Int
    public let supportsMmap: Bool
    public let supportsMlock: Bool
    public let supportsGpuOffload: Bool
    public let supportsRpc: Bool
    public let systemInfo: String
    public let currentTimeMicroseconds: Int64
    
    public init() {
        self.maxDevices = SLlamaSystemInfo.getMaxDevices()
        self.maxParallelSequences = SLlamaSystemInfo.getMaxParallelSequences()
        self.supportsMmap = SLlamaSystemInfo.supportsMmap()
        self.supportsMlock = SLlamaSystemInfo.supportsMlock()
        self.supportsGpuOffload = SLlamaSystemInfo.supportsGpuOffload()
        self.supportsRpc = SLlamaSystemInfo.supportsRpc()
        self.systemInfo = "System information available via logSystemInfo()"
        self.currentTimeMicroseconds = SLlamaSystemInfo.getCurrentTimeMicroseconds()
    }
    
    /// Get a formatted description of system capabilities
    /// - Returns: Formatted string describing system capabilities
    public func getFormattedDescription() -> String {
        var description = "System Capabilities:\n"
        description += "  Max Devices: \(maxDevices)\n"
        description += "  Max Parallel Sequences: \(maxParallelSequences)\n"
        description += "  Memory Mapping: \(supportsMmap ? "Supported" : "Not Supported")\n"
        description += "  Memory Locking: \(supportsMlock ? "Supported" : "Not Supported")\n"
        description += "  GPU Offloading: \(supportsGpuOffload ? "Supported" : "Not Supported")\n"
        description += "  RPC: \(supportsRpc ? "Supported" : "Not Supported")\n"
        description += "  Current Time (μs): \(currentTimeMicroseconds)\n"
        return description
    }
}

// MARK: - Convenience Extensions

public extension SwiftyLlamaCpp {
    
    /// Get comprehensive system information
    /// - Returns: System capabilities information
    static func getSystemCapabilities() -> SLlamaSystemCapabilities {
        return SLlamaSystemCapabilities()
    }
    
    /// Print system information to console
    static func printSystemInfo() {
        let capabilities = getSystemCapabilities()
        print(capabilities.getFormattedDescription())
        print("Detailed System Info:")
        print(capabilities.systemInfo)
    }
} 