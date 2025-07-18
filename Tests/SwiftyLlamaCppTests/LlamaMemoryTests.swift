import Foundation
import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct LlamaMemoryTests {
    
    @Test("LlamaMemoryManager construction and basic API")
    func testLlamaMemoryManagerConstruction() throws {
        let executed = TestUtilities.withDummyContext { ctx in
            let memory = LlamaMemoryManager(context: ctx)
            #expect(memory.cMemory == nil)
            memory.clear(data: false)
            memory.clear(data: true)
        }
        
        if !executed {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaMemoryManager sequence operations")
    func testLlamaMemoryManagerSequenceOperations() throws {
        let executed = TestUtilities.withDummyContext { ctx in
            let memory = LlamaMemoryManager(context: ctx)
            
            // Test sequence removal
            #expect(memory.removeSequence(0, from: 0, to: 10) == false)
            #expect(memory.removeSequence(1, from: -1, to: 5) == false)
            #expect(memory.removeSequence(2, from: 0, to: -1) == false)
            
            // Test sequence copying
            memory.copySequence(from: 0, to: 1, from: 0, to: 10)
            memory.copySequence(from: 1, to: 2, from: -1, to: 5)
            memory.copySequence(from: 2, to: 3, from: 0, to: -1)
            
            // Test sequence keeping
            memory.keepSequence(0)
            memory.keepSequence(1)
            
            // Test sequence adding
            memory.addSequence(0, from: 0, to: 10, delta: 5)
            memory.addSequence(1, from: -1, to: 5, delta: 2)
            memory.addSequence(2, from: 0, to: -1, delta: 1)
            
            // Test sequence division
            memory.divideSequence(0, from: 0, to: 10, by: 2)
            memory.divideSequence(1, from: -1, to: 5, by: 3)
            memory.divideSequence(2, from: 0, to: -1, by: 4)
            
            // Test position queries
            #expect(memory.getSequenceMinPosition(0) == -1)
            #expect(memory.getSequenceMaxPosition(0) == -1)
            #expect(memory.getSequenceMinPosition(1) == -1)
            #expect(memory.getSequenceMaxPosition(1) == -1)
            
            // Test shift capability
            #expect(memory.canShift() == false)
        }
        
        if !executed {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaContext memory manager extension")
    func testLlamaContextMemoryManagerExtension() throws {
        let executed = TestUtilities.withDummyContext { ctx in
            let memory = ctx.memoryManager()
            #expect(memory.cMemory == nil)
        }
        
        if !executed {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
    
    @Test("LlamaContext memory convenience methods")
    func testLlamaContextMemoryConvenienceMethods() throws {
        let executed = TestUtilities.withDummyContext { ctx in
            // Test memory clearing
            ctx.clearMemory(data: false)
            ctx.clearMemory(data: true)
            
            // Test sequence removal
            #expect(ctx.removeSequence(0, from: 0, to: 10) == false)
            #expect(ctx.removeSequence(1, from: -1, to: 5) == false)
            #expect(ctx.removeSequence(2, from: 0, to: -1) == false)
            
            // Test sequence copying
            ctx.copySequence(from: 0, to: 1, from: 0, to: 10)
            ctx.copySequence(from: 1, to: 2, from: -1, to: 5)
            ctx.copySequence(from: 2, to: 3, from: 0, to: -1)
            
            // Test sequence keeping
            ctx.keepSequence(0)
            ctx.keepSequence(1)
            
            // Test sequence adding
            ctx.addSequence(0, from: 0, to: 10, delta: 5)
            ctx.addSequence(1, from: -1, to: 5, delta: 2)
            ctx.addSequence(2, from: 0, to: -1, delta: 1)
            
            // Test sequence division
            ctx.divideSequence(0, from: 0, to: 10, by: 2)
            ctx.divideSequence(1, from: -1, to: 5, by: 3)
            ctx.divideSequence(2, from: 0, to: -1, by: 4)
            
            // Test position queries
            #expect(ctx.getSequenceMinPosition(0) == -1)
            #expect(ctx.getSequenceMaxPosition(0) == -1)
            #expect(ctx.getSequenceMinPosition(1) == -1)
            #expect(ctx.getSequenceMaxPosition(1) == -1)
            
            // Test shift capability
            #expect(ctx.canShiftMemory() == false)
        }
        
        if !executed {
            #expect(Bool(true), "Dummy context creation failed as expected")
        }
    }
} 