import Foundation
import llama

// MARK: - SLlamaState

/// A wrapper for llama state operations
public class SLlamaState {
    // MARK: Properties

    private let context: SLlamaContext

    // MARK: Lifecycle

    /// Initialize with a context
    /// - Parameter context: The llama context to use for state operations
    public init(context: SLlamaContext) {
        self.context = context
    }

    // MARK: Functions

    /// Get the size of the state data
    /// - Returns: The size of the state data in bytes
    public func getStateSize() -> size_t {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_get_size(ctx)
    }

    /// Get state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    /// - Returns: The number of bytes written to the buffer
    public func getStateData(_ data: UnsafeMutablePointer<UInt8>, size: size_t) -> size_t {
        guard let ctx = context.pointer else { return 0 }
        return llama_state_get_data(ctx, data, size)
    }

    /// Set state data
    /// - Parameters:
    ///   - data: Pointer to the data buffer
    ///   - size: Size of the buffer
    /// - Returns: The number of bytes read from the buffer
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
    /// - Returns: True if the state was loaded successfully, false otherwise
    public func loadFromFile(
        _ path: String,
        tokensOut: UnsafeMutablePointer<SLlamaToken>?,
        nTokenCapacity: size_t,
        nTokenCountOut: UnsafeMutablePointer<size_t>?
    )
        -> Bool
    {
        guard let ctx = context.pointer else { return false }
        return llama_state_load_file(ctx, path, tokensOut, nTokenCapacity, nTokenCountOut)
    }

    /// Save state to file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokens: Array of tokens to save
    ///   - nTokenCount: Number of tokens
    /// - Returns: True if the state was saved successfully, false otherwise
    public func saveToFile(
        _ path: String,
        tokens: UnsafePointer<SLlamaToken>,
        nTokenCount: size_t
    )
        -> Bool
    {
        guard let ctx = context.pointer else { return false }
        return llama_state_save_file(ctx, path, tokens, nTokenCount)
    }

    /// Get the size of sequence state data
    /// - Parameter seqId: The sequence ID
    /// - Returns: The size of the sequence state data in bytes
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
    /// - Returns: True if the state was loaded successfully, false otherwise
    func loadStateFromFile(
        _ path: String,
        tokensOut: UnsafeMutablePointer<SLlamaToken>?,
        nTokenCapacity: size_t,
        nTokenCountOut: UnsafeMutablePointer<size_t>?
    )
        -> Bool
    {
        let state = SLlamaState(context: self)
        return state.loadFromFile(path, tokensOut: tokensOut, nTokenCapacity: nTokenCapacity, nTokenCountOut: nTokenCountOut)
    }

    /// Save state to file
    /// - Parameters:
    ///   - path: Path to the state file
    ///   - tokens: Array of tokens to save
    ///   - nTokenCount: Number of tokens
    /// - Returns: True if the state was saved successfully, false otherwise
    func saveStateToFile(
        _ path: String,
        tokens: UnsafePointer<SLlamaToken>,
        nTokenCount: size_t
    )
        -> Bool
    {
        let state = SLlamaState(context: self)
        return state.saveToFile(path, tokens: tokens, nTokenCount: nTokenCount)
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
        return state.loadSequenceStateFromFile(path, destSeqId: destSeqId, tokensOut: tokensOut, nTokenCapacity: nTokenCapacity, nTokenCountOut: nTokenCountOut)
    }

    /// Save complete state to file with error handling
    /// - Parameter path: Path to the state file
    /// - Returns: True if the state was saved successfully, false otherwise
    func saveCompleteState(_ path: String) -> Bool {
        // For complete state save, we pass empty tokens array
        let emptyTokens: [SLlamaToken] = []
        return emptyTokens.withUnsafeBufferPointer { tokens in
            saveStateToFile(path, tokens: tokens.baseAddress!, nTokenCount: tokens.count)
        }
    }

    /// Load complete state from file with error handling
    /// - Parameter path: Path to the state file
    /// - Returns: True if the state was loaded successfully, false otherwise
    func loadCompleteState(_ path: String) -> Bool {
        var tokenCount: size_t = 0
        return loadStateFromFile(path, tokensOut: nil, nTokenCapacity: 0, nTokenCountOut: &tokenCount)
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
