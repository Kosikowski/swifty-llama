import Foundation
import llama

/// A wrapper for llama memory operations
public class LlamaMemoryManager {
    private var memory: LlamaMemory?
    private let context: LlamaContext
    
    /// Initialize with a context
    /// - Parameter context: The llama context to use for memory operations
    public init(context: LlamaContext) {
        self.context = context
        self.memory = llama_get_memory(context.pointer)
    }
    
    /// Get the underlying C memory pointer for direct API access
    public var cMemory: LlamaMemory? {
        return memory
    }
    
    /// Clear all memory
    /// - Parameter data: If true, the data buffers will also be cleared together with the metadata
    public func clear(data: Bool = false) {
        guard let memory = memory else { return }
        llama_memory_clear(memory, data)
    }
    
    /// Remove a sequence from memory
    /// - Parameters:
    ///   - seqId: The sequence ID to remove
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    /// - Returns: True if the sequence was removed, false otherwise
    public func removeSequence(_ seqId: LlamaSequenceId, from p0: LlamaPosition, to p1: LlamaPosition) -> Bool {
        guard let memory = memory else { return false }
        return llama_memory_seq_rm(memory, seqId, p0, p1)
    }
    
    /// Copy a sequence in memory
    /// - Parameters:
    ///   - seqIdSrc: Source sequence ID
    ///   - seqIdDst: Destination sequence ID
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    public func copySequence(
        from seqIdSrc: LlamaSequenceId,
        to seqIdDst: LlamaSequenceId,
        from p0: LlamaPosition,
        to p1: LlamaPosition
    ) {
        guard let memory = memory else { return }
        llama_memory_seq_cp(memory, seqIdSrc, seqIdDst, p0, p1)
    }
    
    /// Keep only a portion of a sequence in memory
    /// - Parameter seqId: The sequence ID
    public func keepSequence(_ seqId: LlamaSequenceId) {
        guard let memory = memory else { return }
        llama_memory_seq_keep(memory, seqId)
    }
    
    /// Add a sequence to memory
    /// - Parameters:
    ///   - seqId: The sequence ID to add
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    ///   - delta: Relative position to add
    public func addSequence(
        _ seqId: LlamaSequenceId,
        from p0: LlamaPosition,
        to p1: LlamaPosition,
        delta: LlamaPosition
    ) {
        guard let memory = memory else { return }
        llama_memory_seq_add(memory, seqId, p0, p1, delta)
    }
    
    /// Divide a sequence in memory
    /// - Parameters:
    ///   - seqId: The sequence ID
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    ///   - d: Division factor (must be > 1)
    public func divideSequence(
        _ seqId: LlamaSequenceId,
        from p0: LlamaPosition,
        to p1: LlamaPosition,
        by d: Int32
    ) {
        guard let memory = memory else { return }
        llama_memory_seq_div(memory, seqId, p0, p1, d)
    }
    
    /// Get the minimum position in a sequence
    /// - Parameter seqId: The sequence ID
    /// - Returns: The minimum position, or -1 if sequence is empty
    public func getSequenceMinPosition(_ seqId: LlamaSequenceId) -> LlamaPosition {
        guard let memory = memory else { return -1 }
        return llama_memory_seq_pos_min(memory, seqId)
    }
    
    /// Get the maximum position in a sequence
    /// - Parameter seqId: The sequence ID
    /// - Returns: The maximum position, or -1 if sequence is empty
    public func getSequenceMaxPosition(_ seqId: LlamaSequenceId) -> LlamaPosition {
        guard let memory = memory else { return -1 }
        return llama_memory_seq_pos_max(memory, seqId)
    }
    
    /// Check if memory can be shifted
    /// - Returns: True if memory can be shifted, false otherwise
    public func canShift() -> Bool {
        guard let memory = memory else { return false }
        return llama_memory_can_shift(memory)
    }
}

// MARK: - Extension to LlamaContext for Memory

public extension LlamaContext {
    
    /// Get memory manager for this context
    /// - Returns: A LlamaMemoryManager instance
    func memoryManager() -> LlamaMemoryManager {
        return LlamaMemoryManager(context: self)
    }
    
    /// Clear all memory for this context
    /// - Parameter data: If true, the data buffers will also be cleared together with the metadata
    func clearMemory(data: Bool = false) {
        let memory = LlamaMemoryManager(context: self)
        memory.clear(data: data)
    }
    
    /// Remove a sequence from memory
    /// - Parameters:
    ///   - seqId: The sequence ID to remove
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    /// - Returns: True if the sequence was removed, false otherwise
    func removeSequence(_ seqId: LlamaSequenceId, from p0: LlamaPosition, to p1: LlamaPosition) -> Bool {
        let memory = LlamaMemoryManager(context: self)
        return memory.removeSequence(seqId, from: p0, to: p1)
    }
    
    /// Copy a sequence in memory
    /// - Parameters:
    ///   - seqIdSrc: Source sequence ID
    ///   - seqIdDst: Destination sequence ID
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    func copySequence(
        from seqIdSrc: LlamaSequenceId,
        to seqIdDst: LlamaSequenceId,
        from p0: LlamaPosition,
        to p1: LlamaPosition
    ) {
        let memory = LlamaMemoryManager(context: self)
        memory.copySequence(from: seqIdSrc, to: seqIdDst, from: p0, to: p1)
    }
    
    /// Keep only a portion of a sequence in memory
    /// - Parameter seqId: The sequence ID
    func keepSequence(_ seqId: LlamaSequenceId) {
        let memory = LlamaMemoryManager(context: self)
        memory.keepSequence(seqId)
    }
    
    /// Add a sequence to memory
    /// - Parameters:
    ///   - seqId: The sequence ID to add
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    ///   - delta: Relative position to add
    func addSequence(
        _ seqId: LlamaSequenceId,
        from p0: LlamaPosition,
        to p1: LlamaPosition,
        delta: LlamaPosition
    ) {
        let memory = LlamaMemoryManager(context: self)
        memory.addSequence(seqId, from: p0, to: p1, delta: delta)
    }
    
    /// Divide a sequence in memory
    /// - Parameters:
    ///   - seqId: The sequence ID
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    ///   - d: Division factor (must be > 1)
    func divideSequence(
        _ seqId: LlamaSequenceId,
        from p0: LlamaPosition,
        to p1: LlamaPosition,
        by d: Int32
    ) {
        let memory = LlamaMemoryManager(context: self)
        memory.divideSequence(seqId, from: p0, to: p1, by: d)
    }
    
    /// Get the minimum position in a sequence
    /// - Parameter seqId: The sequence ID
    /// - Returns: The minimum position, or -1 if sequence is empty
    func getSequenceMinPosition(_ seqId: LlamaSequenceId) -> LlamaPosition {
        let memory = LlamaMemoryManager(context: self)
        return memory.getSequenceMinPosition(seqId)
    }
    
    /// Get the maximum position in a sequence
    /// - Parameter seqId: The sequence ID
    /// - Returns: The maximum position, or -1 if sequence is empty
    func getSequenceMaxPosition(_ seqId: LlamaSequenceId) -> LlamaPosition {
        let memory = LlamaMemoryManager(context: self)
        return memory.getSequenceMaxPosition(seqId)
    }
    
    /// Check if memory can be shifted
    /// - Returns: True if memory can be shifted, false otherwise
    func canShiftMemory() -> Bool {
        let memory = LlamaMemoryManager(context: self)
        return memory.canShift()
    }
} 