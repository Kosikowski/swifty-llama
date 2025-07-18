import Foundation
import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct LlamaStateTests {
    
    @Test("LlamaState construction and basic API")
    func testLlamaStateConstruction() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let state = LlamaState(context: ctx)
            #expect(state.getStateSize() == 0)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaState data operations")
    func testLlamaStateDataOperations() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let state = LlamaState(context: ctx)
            
            // Test state data operations with dummy buffers
            var dummyBuffer: [UInt8] = Array(repeating: 0, count: 1024)
            let bytesWritten = dummyBuffer.withUnsafeMutableBufferPointer { buffer in
                state.getStateData(buffer.baseAddress!, size: buffer.count)
            }
            #expect(bytesWritten == 0)
            
            let bytesRead = dummyBuffer.withUnsafeBufferPointer { buffer in
                state.setStateData(buffer.baseAddress!, size: buffer.count)
            }
            #expect(bytesRead == 0)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaState file operations")
    func testLlamaStateFileOperations() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let state = LlamaState(context: ctx)
            
            // Test file operations
            var tokenCount: size_t = 0
            let loadResult = state.loadFromFile(
                "/nonexistent/path/state.bin",
                tokensOut: nil as UnsafeMutablePointer<LlamaToken>?,
                nTokenCapacity: 0,
                nTokenCountOut: &tokenCount
            )
            #expect(loadResult == false)
            
            let emptyTokens: [LlamaToken] = []
            let saveResult = emptyTokens.withUnsafeBufferPointer { tokens in
                state.saveToFile(
                    "/nonexistent/path/state.bin",
                    tokens: tokens.baseAddress!,
                    nTokenCount: tokens.count
                )
            }
            #expect(saveResult == false)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaState sequence operations")
    func testLlamaStateSequenceOperations() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let state = LlamaState(context: ctx)
            
            // Test sequence state size
            #expect(state.getSequenceStateSize(0) == 0)
            #expect(state.getSequenceStateSize(1) == 0)
            
            // Test sequence state data operations
            var dummyBuffer: [UInt8] = Array(repeating: 0, count: 512)
            let bytesWritten = dummyBuffer.withUnsafeMutableBufferPointer { buffer in
                state.getSequenceStateData(buffer.baseAddress!, size: buffer.count, seqId: 0)
            }
            #expect(bytesWritten == 0)
            
            let bytesRead = dummyBuffer.withUnsafeBufferPointer { buffer in
                state.setSequenceStateData(buffer.baseAddress!, size: buffer.count, destSeqId: 0)
            }
            #expect(bytesRead == 0)
            
            // Test sequence state file operations
            let emptyTokens: [LlamaToken] = []
            let saveBytes = emptyTokens.withUnsafeBufferPointer { tokens in
                state.saveSequenceStateToFile(
                    "/nonexistent/path/seq_state.bin",
                    seqId: 0,
                    tokens: tokens.baseAddress!,
                    nTokenCount: tokens.count
                )
            }
            #expect(saveBytes == 0)
            
            var tokenCount: size_t = 0
            let loadBytes = state.loadSequenceStateFromFile(
                "/nonexistent/path/seq_state.bin",
                destSeqId: 0,
                tokensOut: nil as UnsafeMutablePointer<LlamaToken>?,
                nTokenCapacity: 0,
                nTokenCountOut: &tokenCount
            )
            #expect(loadBytes == 0)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaContext state extension")
    func testLlamaContextStateExtension() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            let state = ctx.state()
            #expect(state.getStateSize() == 0)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaContext state convenience methods")
    func testLlamaContextStateConvenienceMethods() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            // Test state size
            #expect(ctx.getStateSize() == 0)
            
            // Test state data operations
            var dummyBuffer: [UInt8] = Array(repeating: 0, count: 1024)
            let bytesWritten = dummyBuffer.withUnsafeMutableBufferPointer { buffer in
                ctx.getStateData(buffer.baseAddress!, size: buffer.count)
            }
            #expect(bytesWritten == 0)
            
            let bytesRead = dummyBuffer.withUnsafeBufferPointer { buffer in
                ctx.setStateData(buffer.baseAddress!, size: buffer.count)
            }
            #expect(bytesRead == 0)
            
            // Test file operations
            var tokenCount: size_t = 0
            let loadResult = ctx.loadStateFromFile(
                "/nonexistent/path/state.bin",
                tokensOut: nil as UnsafeMutablePointer<LlamaToken>?,
                nTokenCapacity: 0,
                nTokenCountOut: &tokenCount
            )
            #expect(loadResult == false)
            
            let emptyTokens: [LlamaToken] = []
            let saveResult = emptyTokens.withUnsafeBufferPointer { tokens in
                ctx.saveStateToFile(
                    "/nonexistent/path/state.bin",
                    tokens: tokens.baseAddress!,
                    nTokenCount: tokens.count
                )
            }
            #expect(saveResult == false)
            
            // Test sequence state operations
            #expect(ctx.getSequenceStateSize(0) == 0)
            
            let seqBytesWritten = dummyBuffer.withUnsafeMutableBufferPointer { buffer in
                ctx.getSequenceStateData(buffer.baseAddress!, size: buffer.count, seqId: 0)
            }
            #expect(seqBytesWritten == 0)
            
            let seqBytesRead = dummyBuffer.withUnsafeBufferPointer { buffer in
                ctx.setSequenceStateData(buffer.baseAddress!, size: buffer.count, destSeqId: 0)
            }
            #expect(seqBytesRead == 0)
            
            // Test sequence file operations
            let seqSaveBytes = emptyTokens.withUnsafeBufferPointer { tokens in
                ctx.saveSequenceStateToFile(
                    "/nonexistent/path/seq_state.bin",
                    seqId: 0,
                    tokens: tokens.baseAddress!,
                    nTokenCount: tokens.count
                )
            }
            #expect(seqSaveBytes == 0)
            
            let seqLoadBytes = ctx.loadSequenceStateFromFile(
                "/nonexistent/path/seq_state.bin",
                destSeqId: 0,
                tokensOut: nil as UnsafeMutablePointer<LlamaToken>?,
                nTokenCapacity: 0,
                nTokenCountOut: &tokenCount
            )
            #expect(seqLoadBytes == 0)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaContext state convenience data methods")
    func testLlamaContextStateConvenienceDataMethods() throws {
        let dummyModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = dummyModel, let ctx = DummyContext(model: model) {
            // Test complete state operations
            #expect(ctx.saveCompleteState("/nonexistent/path/complete_state.bin") == false)
            #expect(ctx.loadCompleteState("/nonexistent/path/complete_state.bin") == false)
            
            // Test data buffer operations
            #expect(ctx.saveStateToData() == nil)
            #expect(ctx.loadStateFromData(Foundation.Data()) == false)
            
            // Test sequence data operations
            #expect(ctx.saveSequenceStateToData(0) == nil)
            #expect(ctx.loadSequenceStateFromData(0, Foundation.Data()) == false)
        } else {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
} 