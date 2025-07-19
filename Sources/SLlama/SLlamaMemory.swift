import Foundation
import llama

// MARK: - SLlamaMemoryManager

/// A wrapper for llama memory operations
public class SLlamaMemoryManager {
    // MARK: Properties

    private var memory: SLlamaMemory?
    private let context: SLlamaContext

    // MARK: Computed Properties

    /// Get the underlying C memory pointer for direct API access
    public var cMemory: SLlamaMemory? {
        memory
    }

    // MARK: Lifecycle

    /// Initialize with a context
    /// - Parameter context: The llama context to use for memory operations
    public init(context: SLlamaContext) {
        self.context = context
        memory = llama_get_memory(context.pointer)
    }

    // MARK: Functions

    /// Clear all memory
    /// - Parameter data: If true, the data buffers will also be cleared together with the metadata
    public func clear(data: Bool = false) {
        guard let memory else { return }
        llama_memory_clear(memory, data)
    }

    /// Remove a sequence from memory
    /// - Parameters:
    ///   - seqId: The sequence ID to remove
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    /// - Returns: True if the sequence was removed, false otherwise
    public func removeSequence(_ seqId: SLlamaSequenceId, from p0: SLlamaPosition, to p1: SLlamaPosition) -> Bool {
        guard let memory else { return false }
        return llama_memory_seq_rm(memory, seqId, p0, p1)
    }

    /// Copy a sequence in memory
    /// - Parameters:
    ///   - seqIdSrc: Source sequence ID
    ///   - seqIdDst: Destination sequence ID
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    public func copySequence(
        from seqIdSrc: SLlamaSequenceId,
        to seqIdDst: SLlamaSequenceId,
        from p0: SLlamaPosition,
        to p1: SLlamaPosition
    ) {
        guard let memory else { return }
        llama_memory_seq_cp(memory, seqIdSrc, seqIdDst, p0, p1)
    }

    /// Keep only a portion of a sequence in memory
    /// - Parameter seqId: The sequence ID
    public func keepSequence(_ seqId: SLlamaSequenceId) {
        guard let memory else { return }
        llama_memory_seq_keep(memory, seqId)
    }

    /// Add a portion of a sequence to memory
    /// - Parameters:
    ///   - seqId: The sequence ID
    ///   - p0: Start position
    ///   - p1: End position
    ///   - delta: Relative position to add
    public func addSequence(
        _ seqId: SLlamaSequenceId,
        from p0: SLlamaPosition,
        to p1: SLlamaPosition,
        delta: SLlamaPosition
    ) {
        guard let memory else { return }
        llama_memory_seq_add(memory, seqId, p0, p1, delta)
    }

    /// Divide a sequence in memory
    /// - Parameters:
    ///   - seqId: The sequence ID
    ///   - p0: Start position
    ///   - p1: End position
    ///   - d: Division factor (must be > 1)
    public func divideSequence(
        _ seqId: SLlamaSequenceId,
        from p0: SLlamaPosition,
        to p1: SLlamaPosition,
        by d: Int32
    ) {
        guard let memory else { return }
        llama_memory_seq_div(memory, seqId, p0, p1, d)
    }

    /// Get the minimum position in a sequence
    /// - Parameter seqId: The sequence ID
    /// - Returns: The minimum position, or -1 if sequence is empty
    public func getSequenceMinPosition(_ seqId: SLlamaSequenceId) -> SLlamaPosition {
        guard let memory else { return -1 }
        return llama_memory_seq_pos_min(memory, seqId)
    }

    /// Get the maximum position in a sequence
    /// - Parameter seqId: The sequence ID
    /// - Returns: The maximum position, or -1 if sequence is empty
    public func getSequenceMaxPosition(_ seqId: SLlamaSequenceId) -> SLlamaPosition {
        guard let memory else { return -1 }
        return llama_memory_seq_pos_max(memory, seqId)
    }

    /// Check if memory can be shifted
    /// - Returns: True if memory can be shifted, false otherwise
    public func canShift() -> Bool {
        guard let memory else { return false }
        return llama_memory_can_shift(memory)
    }
}

// MARK: - Extension to SLlamaContext for Memory

public extension SLlamaContext {
    /// Get memory manager for this context
    /// - Returns: A SLlamaMemoryManager instance
    func memoryManager() -> SLlamaMemoryManager {
        SLlamaMemoryManager(context: self)
    }

    /// Clear all memory
    /// - Parameter data: If true, the data buffers will also be cleared together with the metadata
    func clearMemory(data: Bool = false) {
        memoryManager().clear(data: data)
    }

    /// Remove a sequence from memory
    /// - Parameters:
    ///   - seqId: The sequence ID to remove
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    /// - Returns: True if the sequence was removed, false otherwise
    func removeSequence(_ seqId: SLlamaSequenceId, from p0: SLlamaPosition, to p1: SLlamaPosition) -> Bool {
        let memory = SLlamaMemoryManager(context: self)
        return memory.removeSequence(seqId, from: p0, to: p1)
    }

    /// Copy a sequence in memory
    /// - Parameters:
    ///   - seqIdSrc: Source sequence ID
    ///   - seqIdDst: Destination sequence ID
    ///   - p0: Start position (p0 < 0 means [0, p1])
    ///   - p1: End position (p1 < 0 means [p0, inf))
    func copySequence(
        from seqIdSrc: SLlamaSequenceId,
        to seqIdDst: SLlamaSequenceId,
        from p0: SLlamaPosition,
        to p1: SLlamaPosition
    ) {
        let memory = SLlamaMemoryManager(context: self)
        memory.copySequence(from: seqIdSrc, to: seqIdDst, from: p0, to: p1)
    }

    /// Keep only a portion of a sequence in memory
    /// - Parameter seqId: The sequence ID
    func keepSequence(_ seqId: SLlamaSequenceId) {
        let memory = SLlamaMemoryManager(context: self)
        memory.keepSequence(seqId)
    }

    /// Add a portion of a sequence to memory
    /// - Parameters:
    ///   - seqId: The sequence ID
    ///   - p0: Start position
    ///   - p1: End position
    ///   - delta: Relative position to add
    func addSequence(
        _ seqId: SLlamaSequenceId,
        from p0: SLlamaPosition,
        to p1: SLlamaPosition,
        delta: SLlamaPosition
    ) {
        let memory = SLlamaMemoryManager(context: self)
        memory.addSequence(seqId, from: p0, to: p1, delta: delta)
    }

    /// Divide a sequence in memory
    /// - Parameters:
    ///   - seqId: The sequence ID
    ///   - p0: Start position
    ///   - p1: End position
    ///   - d: Division factor (must be > 1)
    func divideSequence(
        _ seqId: SLlamaSequenceId,
        from p0: SLlamaPosition,
        to p1: SLlamaPosition,
        by d: Int32
    ) {
        let memory = SLlamaMemoryManager(context: self)
        memory.divideSequence(seqId, from: p0, to: p1, by: d)
    }

    /// Get the minimum position in a sequence
    /// - Parameter seqId: The sequence ID
    /// - Returns: The minimum position, or -1 if sequence is empty
    func getSequenceMinPosition(_ seqId: SLlamaSequenceId) -> SLlamaPosition {
        let memory = SLlamaMemoryManager(context: self)
        return memory.getSequenceMinPosition(seqId)
    }

    /// Get the maximum position in a sequence
    /// - Parameter seqId: The sequence ID
    /// - Returns: The maximum position, or -1 if sequence is empty
    func getSequenceMaxPosition(_ seqId: SLlamaSequenceId) -> SLlamaPosition {
        let memory = SLlamaMemoryManager(context: self)
        return memory.getSequenceMaxPosition(seqId)
    }

    /// Check if memory can be shifted
    /// - Returns: True if memory can be shifted, false otherwise
    func canShiftMemory() -> Bool {
        let memory = SLlamaMemoryManager(context: self)
        return memory.canShift()
    }
}
