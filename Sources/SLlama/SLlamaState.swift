import Foundation
import llama

// MARK: - SLlamaState

/// A wrapper for llama state operations
public class SLlamaState: @unchecked Sendable {
    // MARK: Properties

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    let context: SLlamaContext

    // MARK: Lifecycle

    /// Initialize with a context
    /// - Parameter context: The llama context to use for state operations
    public init(context: SLlamaContext) {
        self.context = context
    }

    // MARK: Functions

    /// Get the size of the state data
    /// - Returns: The size of the state data in bytes
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getStateSize() -> size_t {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_get_size(ctx)
    }

    /// Get state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    /// - Returns: The number of bytes written to the buffer
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getStateData(_ data: UnsafeMutablePointer<UInt8>, size: size_t) -> size_t {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_get_data(ctx, data, size)
    }

    /// Set state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    /// - Returns: The number of bytes read from the buffer
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setStateData(_ data: UnsafePointer<UInt8>, size: size_t) -> size_t {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_set_data(ctx, data, size)
    }

    /// Load state from file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokensOut: Buffer for tokens
    ///   - nTokenCapacity: Capacity of the token buffer
    ///   - nTokenCountOut: Output parameter for number of tokens
    /// - Throws: SLlamaError if loading fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func loadFromFile(
        _ path: String,
        tokensOut: UnsafeMutablePointer<SLlamaToken>?,
        nTokenCapacity: size_t,
        nTokenCountOut: UnsafeMutablePointer<size_t>?
    ) throws {
        guard let ctx = context.pointer else {
            throw SLlamaError.contextNotInitialized
        }

        // Validate file exists and is readable
        guard FileManager.default.fileExists(atPath: path) else {
            throw SLlamaError.fileNotFound(path)
        }

        guard FileManager.default.isReadableFile(atPath: path) else {
            throw SLlamaError.permissionDenied(path)
        }

        // Check file size
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: path)
            let fileSize = attributes[.size] as? Int64 ?? 0
            if fileSize == 0 {
                throw SLlamaError.corruptedFile("State file is empty: '\(path)'")
            }
        } catch {
            throw SLlamaError
                .fileAccessError("Could not read state file attributes for '\(path)': \(error.localizedDescription)")
        }

        let success = llama_state_load_file(ctx, path, tokensOut, nTokenCapacity, nTokenCountOut)

        guard success else {
            // Try to determine specific error
            if let nTokenCountOut = nTokenCountOut?.pointee, nTokenCountOut > nTokenCapacity {
                throw SLlamaError.bufferTooSmall
            }
            throw SLlamaError
                .stateLoadingFailed("Could not load state from '\(path)' (file may be corrupted or incompatible)")
        }
    }

    /// Save state to file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokens: Array of tokens to save
    ///   - nTokenCount: Number of tokens
    /// - Throws: SLlamaError if saving fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func saveToFile(
        _ path: String,
        tokens: UnsafePointer<SLlamaToken>,
        nTokenCount: size_t
    ) throws {
        guard let ctx = context.pointer else {
            throw SLlamaError.contextNotInitialized
        }

        // Validate output directory exists and is writable
        let fileURL = URL(fileURLWithPath: path)
        let directory = fileURL.deletingLastPathComponent().path

        guard FileManager.default.fileExists(atPath: directory) else {
            throw SLlamaError.fileNotFound("Directory does not exist: \(directory)")
        }

        guard FileManager.default.isWritableFile(atPath: directory) else {
            throw SLlamaError.permissionDenied("Cannot write to directory: \(directory)")
        }

        // Check available disk space
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: directory)
            let availableSpace = attributes[.systemFreeSize] as? Int64 ?? 0

            // Rough estimate: state size + token array size
            let stateSize = getStateSize()
            let tokenSize = nTokenCount * MemoryLayout<SLlamaToken>.size
            let totalSize = Int64(stateSize + tokenSize)

            if availableSpace < totalSize {
                throw SLlamaError.insufficientSpace
            }
        } catch {
            throw SLlamaError
                .fileAccessError(
                    "Could not check disk space for directory '\(directory)': \(error.localizedDescription)"
                )
        }

        let success = llama_state_save_file(ctx, path, tokens, nTokenCount)

        guard success else {
            throw SLlamaError
                .stateSavingFailed("Could not save state to '\(path)' (check file permissions and disk space)")
        }
    }

    /// Legacy load method that returns bool (deprecated)
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokensOut: Buffer for tokens
    ///   - nTokenCapacity: Capacity of the token buffer
    ///   - nTokenCountOut: Output parameter for number of tokens
    /// - Returns: True if the state was loaded successfully, false otherwise
    @available(*, deprecated, message: "Use loadFromFile(_:tokensOut:nTokenCapacity:nTokenCountOut:) throws instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func _loadFromFile(
        _ path: String,
        tokensOut: UnsafeMutablePointer<SLlamaToken>?,
        nTokenCapacity: size_t,
        nTokenCountOut: UnsafeMutablePointer<size_t>?
    )
        -> Bool
    {
        do {
            try loadFromFile(path, tokensOut: tokensOut, nTokenCapacity: nTokenCapacity, nTokenCountOut: nTokenCountOut)
            return true
        } catch {
            return false
        }
    }

    /// Legacy save method that returns bool (deprecated)
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokens: Array of tokens to save
    ///   - nTokenCount: Number of tokens
    /// - Returns: True if the state was saved successfully, false otherwise
    @available(*, deprecated, message: "Use saveToFile(_:tokens:nTokenCount:) throws instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func _saveToFile(
        _ path: String,
        tokens: UnsafePointer<SLlamaToken>,
        nTokenCount: size_t
    )
        -> Bool
    {
        do {
            try saveToFile(path, tokens: tokens, nTokenCount: nTokenCount)
            return true
        } catch {
            return false
        }
    }

    /// Get the size of sequence state data
    /// - Parameter seqId: The sequence ID
    /// - Returns: The size of the sequence state data in bytes
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getSequenceStateSize(_ seqId: SLlamaSequenceId) -> size_t {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_seq_get_size(ctx, seqId)
    }

    /// Get sequence state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    ///   - seqId: The sequence ID
    /// - Returns: The number of bytes written to the buffer
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getSequenceStateData(
        _ data: UnsafeMutablePointer<UInt8>,
        size: size_t,
        seqId: SLlamaSequenceId
    )
        -> size_t
    {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_seq_get_data(ctx, data, size, seqId)
    }

    /// Set sequence state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    ///   - destSeqId: Destination sequence ID
    /// - Returns: The number of bytes read from the buffer
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setSequenceStateData(
        _ data: UnsafePointer<UInt8>,
        size: size_t,
        destSeqId: SLlamaSequenceId
    )
        -> size_t
    {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_seq_set_data(ctx, data, size, destSeqId)
    }

    /// Save sequence state to file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - seqId: The sequence ID
    ///   - tokens: Array of tokens to save
    ///   - nTokenCount: Number of tokens
    /// - Returns: The number of bytes written to the file
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func saveSequenceStateToFile(
        _ path: String,
        seqId: SLlamaSequenceId,
        tokens: UnsafePointer<SLlamaToken>,
        nTokenCount: size_t
    )
        -> size_t
    {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_seq_save_file(ctx, path, seqId, tokens, nTokenCount)
    }

    /// Load sequence state from file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - destSeqId: Destination sequence ID
    ///   - tokensOut: Buffer for tokens
    ///   - nTokenCapacity: Capacity of the token buffer
    ///   - nTokenCountOut: Output parameter for number of tokens
    /// - Returns: The number of bytes read from the file
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func loadSequenceStateFromFile(
        _ path: String,
        destSeqId: SLlamaSequenceId,
        tokensOut: UnsafeMutablePointer<SLlamaToken>?,
        nTokenCapacity: size_t,
        nTokenCountOut: UnsafeMutablePointer<size_t>?
    )
        -> size_t
    {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_seq_load_file(ctx, path, destSeqId, tokensOut, nTokenCapacity, nTokenCountOut)
    }
}

// MARK: - Extension to SLlamaContext for State

/// Extension to SLlamaContext for state operations
public extension SLlamaContext {
    /// Get state for this context
    /// - Returns: A SLlamaState instance
    func state() -> SLlamaState {
        SLlamaState(context: self)
    }

    /// Get the size of the state data
    /// - Returns: The size of the state data in bytes
    func getStateSize() -> size_t {
        let state = SLlamaState(context: self)
        return state.getStateSize()
    }

    /// Get state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    /// - Returns: The number of bytes written to the buffer
    func getStateData(_ data: UnsafeMutablePointer<UInt8>, size: size_t) -> size_t {
        let state = SLlamaState(context: self)
        return state.getStateData(data, size: size)
    }

    /// Set state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    /// - Returns: The number of bytes read from the buffer
    func setStateData(_ data: UnsafePointer<UInt8>, size: size_t) -> size_t {
        let state = SLlamaState(context: self)
        return state.setStateData(data, size: size)
    }

    /// Load state from file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokensOut: Buffer for tokens
    ///   - nTokenCapacity: Capacity of the token buffer
    ///   - nTokenCountOut: Output parameter for number of tokens
    /// - Throws: SLlamaError if loading fails
    func loadStateFromFile(
        _ path: String,
        tokensOut: UnsafeMutablePointer<SLlamaToken>?,
        nTokenCapacity: size_t,
        nTokenCountOut: UnsafeMutablePointer<size_t>?
    ) throws {
        let state = SLlamaState(context: self)
        try state.loadFromFile(
            path,
            tokensOut: tokensOut,
            nTokenCapacity: nTokenCapacity,
            nTokenCountOut: nTokenCountOut
        )
    }

    /// Save state to file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokens: Array of tokens to save
    ///   - nTokenCount: Number of tokens
    /// - Throws: SLlamaError if saving fails
    func saveStateToFile(
        _ path: String,
        tokens: UnsafePointer<SLlamaToken>,
        nTokenCount: size_t
    ) throws {
        let state = SLlamaState(context: self)
        try state.saveToFile(path, tokens: tokens, nTokenCount: nTokenCount)
    }

    // MARK: - Legacy Methods (Deprecated)

    /// Legacy load state method that returns bool (deprecated)
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokensOut: Buffer for tokens
    ///   - nTokenCapacity: Capacity of the token buffer
    ///   - nTokenCountOut: Output parameter for number of tokens
    /// - Returns: True if the state was loaded successfully, false otherwise
    @available(
        *,
        deprecated,
        message: "Use loadStateFromFile(_:tokensOut:nTokenCapacity:nTokenCountOut:) throws instead"
    )
    func _loadStateFromFile(
        _ path: String,
        tokensOut: UnsafeMutablePointer<SLlamaToken>?,
        nTokenCapacity: size_t,
        nTokenCountOut: UnsafeMutablePointer<size_t>?
    )
        -> Bool
    {
        do {
            try loadStateFromFile(
                path,
                tokensOut: tokensOut,
                nTokenCapacity: nTokenCapacity,
                nTokenCountOut: nTokenCountOut
            )
            return true
        } catch {
            return false
        }
    }

    /// Legacy save state method that returns bool (deprecated)
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokens: Array of tokens to save
    ///   - nTokenCount: Number of tokens
    /// - Returns: True if the state was saved successfully, false otherwise
    @available(*, deprecated, message: "Use saveStateToFile(_:tokens:nTokenCount:) throws instead")
    func _saveStateToFile(
        _ path: String,
        tokens: UnsafePointer<SLlamaToken>,
        nTokenCount: size_t
    )
        -> Bool
    {
        do {
            try saveStateToFile(path, tokens: tokens, nTokenCount: nTokenCount)
            return true
        } catch {
            return false
        }
    }

    /// Get the size of sequence state data
    /// - Parameter seqId: The sequence ID
    /// - Returns: The size of the sequence state data in bytes
    func getSequenceStateSize(_ seqId: SLlamaSequenceId) -> size_t {
        let state = SLlamaState(context: self)
        return state.getSequenceStateSize(seqId)
    }

    /// Get sequence state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    ///   - seqId: The sequence ID
    /// - Returns: The number of bytes written to the buffer
    func getSequenceStateData(
        _ data: UnsafeMutablePointer<UInt8>,
        size: size_t,
        seqId: SLlamaSequenceId
    )
        -> size_t
    {
        let state = SLlamaState(context: self)
        return state.getSequenceStateData(data, size: size, seqId: seqId)
    }

    /// Set sequence state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    ///   - destSeqId: Destination sequence ID
    /// - Returns: The number of bytes read from the buffer
    func setSequenceStateData(
        _ data: UnsafePointer<UInt8>,
        size: size_t,
        destSeqId: SLlamaSequenceId
    )
        -> size_t
    {
        let state = SLlamaState(context: self)
        return state.setSequenceStateData(data, size: size, destSeqId: destSeqId)
    }

    /// Save sequence state to file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - seqId: The sequence ID
    ///   - tokens: Array of tokens to save
    ///   - nTokenCount: Number of tokens
    /// - Returns: The number of bytes written to the file
    func saveSequenceStateToFile(
        _ path: String,
        seqId: SLlamaSequenceId,
        tokens: UnsafePointer<SLlamaToken>,
        nTokenCount: size_t
    )
        -> size_t
    {
        let state = SLlamaState(context: self)
        return state.saveSequenceStateToFile(path, seqId: seqId, tokens: tokens, nTokenCount: nTokenCount)
    }

    /// Load sequence state from file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - destSeqId: Destination sequence ID
    ///   - tokensOut: Buffer for tokens
    ///   - nTokenCapacity: Capacity of the token buffer
    ///   - nTokenCountOut: Output parameter for number of tokens
    /// - Returns: The number of bytes read from the file
    func loadSequenceStateFromFile(
        _ path: String,
        destSeqId: SLlamaSequenceId,
        tokensOut: UnsafeMutablePointer<SLlamaToken>?,
        nTokenCapacity: size_t,
        nTokenCountOut: UnsafeMutablePointer<size_t>?
    )
        -> size_t
    {
        let state = SLlamaState(context: self)
        return state.loadSequenceStateFromFile(
            path,
            destSeqId: destSeqId,
            tokensOut: tokensOut,
            nTokenCapacity: nTokenCapacity,
            nTokenCountOut: nTokenCountOut
        )
    }

    /// Save complete state to file
    /// - Parameter path: Path to the state file
    /// - Throws: SLlamaError if saving fails
    func saveCompleteState(_ path: String) throws {
        // For complete state save, we pass empty tokens array
        let emptyTokens: [SLlamaToken] = []
        try emptyTokens.withUnsafeBufferPointer { tokens in
            try saveStateToFile(path, tokens: tokens.baseAddress!, nTokenCount: tokens.count)
        }
    }

    /// Load complete state from file
    /// - Parameter path: Path to the state file
    /// - Throws: SLlamaError if loading fails
    func loadCompleteState(_ path: String) throws {
        var tokenCount: size_t = 0
        try loadStateFromFile(path, tokensOut: nil, nTokenCapacity: 0, nTokenCountOut: &tokenCount)
    }

    /// Legacy save complete state method that returns bool (deprecated)
    /// - Parameter path: Path to the state file
    /// - Returns: True if the state was saved successfully, false otherwise
    @available(*, deprecated, message: "Use saveCompleteState(_:) throws instead")
    func _saveCompleteState(_ path: String) -> Bool {
        do {
            try saveCompleteState(path)
            return true
        } catch {
            return false
        }
    }

    /// Legacy load complete state method that returns bool (deprecated)
    /// - Parameter path: Path to the state file
    /// - Returns: True if the state was loaded successfully, false otherwise
    @available(*, deprecated, message: "Use loadCompleteState(_:) throws instead")
    func _loadCompleteState(_ path: String) -> Bool {
        do {
            try loadCompleteState(path)
            return true
        } catch {
            return false
        }
    }

    /// Save state data to buffer
    /// - Returns: The state data as Data, or nil if failed
    func saveStateToData() -> Data? {
        let size = getStateSize()
        guard size > 0 else { return nil }

        var data = Data(count: size)
        let bytesWritten = data.withUnsafeMutableBytes { buffer in
            getStateData(buffer.bindMemory(to: UInt8.self).baseAddress!, size: size)
        }

        guard bytesWritten == size else { return nil }
        return data
    }

    /// Load state data from buffer
    /// - Parameter data: The state data
    /// - Returns: True if the state was loaded successfully, false otherwise
    func loadStateFromData(_ data: Data) -> Bool {
        let bytesRead = data.withUnsafeBytes { buffer in
            setStateData(buffer.bindMemory(to: UInt8.self).baseAddress!, size: data.count)
        }
        return bytesRead == data.count
    }

    /// Save sequence state data to buffer
    /// - Parameter seqId: The sequence ID
    /// - Returns: The sequence state data as Data, or nil if failed
    func saveSequenceStateToData(_ seqId: SLlamaSequenceId) -> Data? {
        let size = getSequenceStateSize(seqId)
        guard size > 0 else { return nil }

        var data = Data(count: size)
        let bytesWritten = data.withUnsafeMutableBytes { buffer in
            getSequenceStateData(buffer.bindMemory(to: UInt8.self).baseAddress!, size: size, seqId: seqId)
        }

        guard bytesWritten == size else { return nil }
        return data
    }

    /// Load sequence state data from buffer
    /// - Parameters:
    ///   - seqId: The sequence ID
    ///   - data: The sequence state data
    /// - Returns: True if the state was loaded successfully, false otherwise
    func loadSequenceStateFromData(_ seqId: SLlamaSequenceId, _ data: Data) -> Bool {
        let bytesRead = data.withUnsafeBytes { buffer in
            setSequenceStateData(buffer.bindMemory(to: UInt8.self).baseAddress!, size: data.count, destSeqId: seqId)
        }
        return bytesRead == data.count
    }
}
