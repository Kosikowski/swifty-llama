import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct LlamaTypesTests {
    
    @Test("Type aliases are properly defined")
    func testTypeAliases() throws {
        // Test that type aliases are properly defined
        #expect(type(of: LlamaTokenNull) == LlamaToken.self, "LlamaTokenNull should be LlamaToken type")
        #expect(type(of: LlamaDefaultSeed) == UInt32.self, "LlamaDefaultSeed should be UInt32 type")
        #expect(type(of: LlamaFileMagicGGLA) == UInt32.self, "LlamaFileMagicGGLA should be UInt32 type")
    }
    
    @Test("Enum type aliases work correctly")
    func testEnumTypeAliases() throws {
        // Test enum type aliases exist and have correct types
        #expect(LlamaVocabType.none.rawValue == 0, "LlamaVocabType.none should have raw value 0")
        #expect(LlamaRopeType.norm.rawValue == 0, "LlamaRopeType.norm should have raw value 0")
        #expect(LlamaTokenType.normal.rawValue == 1, "LlamaTokenType.normal should have raw value 1")
        #expect(LlamaFileType.mostlyQ4_0.rawValue == 2, "LlamaFileType.mostlyQ4_0 should have raw value 2")
    }
    
    @Test("Basic type definitions")
    func testBasicTypeDefinitions() throws {
        // Test that basic C types are properly bridged
        #expect(type(of: LlamaTokenNull) == LlamaToken.self, "LlamaTokenNull should be LlamaToken")
        #expect(type(of: LlamaVocabType.none) == LlamaVocabType.self, "LlamaVocabType.none should be LlamaVocabType")
        #expect(type(of: LlamaTokenAttribute.undefined) == LlamaTokenAttribute.self, "LlamaTokenAttribute.undefined should be LlamaTokenAttribute")
    }
    
    @Test("Pointer type aliases")
    func testPointerTypeAliases() throws {
        // Test that pointer type aliases are properly defined
        #expect(LlamaModelPointer.self == OpaquePointer.self, "LlamaModelPointer should be OpaquePointer")
        #expect(LlamaContextPointer.self == OpaquePointer.self, "LlamaContextPointer should be OpaquePointer")
        #expect(LlamaVocabPointer.self == OpaquePointer.self, "LlamaVocabPointer should be OpaquePointer")
    }
    
    @Test("Constants and magic values")
    func testConstantsAndMagicValues() throws {
        // Test that constants are properly defined
        #expect(LlamaTokenNull == -1, "LlamaTokenNull should be -1")
        #expect(LlamaDefaultSeed == 0xFFFFFFFF, "LlamaDefaultSeed should be 0xFFFFFFFF")
        #expect(LlamaFileMagicGGLA == 0x67676C61, "LlamaFileMagicGGLA should be correct magic")
        #expect(LlamaFileMagicGGSN == 0x6767736E, "LlamaFileMagicGGSN should be correct magic")
        #expect(LlamaFileMagicGGSQ == 0x67677371, "LlamaFileMagicGGSQ should be correct magic")
        #expect(LlamaSessionMagic == 0x6767736E, "LlamaSessionMagic should be correct magic")
        #expect(LlamaSessionVersion == 9, "LlamaSessionVersion should be 9")
        #expect(LlamaStateSeqMagic == 0x67677371, "LlamaStateSeqMagic should be correct magic")
        #expect(LlamaStateSeqVersion == 2, "LlamaStateSeqVersion should be 2")
    }
    
    @Test("Enum extensions")
    func testEnumExtensions() throws {
        // Test that enum extensions provide correct values
        #expect(LlamaVocabType.none.rawValue == 0, "LlamaVocabType.none should be 0")
        #expect(LlamaVocabType.spm.rawValue == 1, "LlamaVocabType.spm should be 1")
        #expect(LlamaVocabType.bpe.rawValue == 2, "LlamaVocabType.bpe should be 2")
        
        #expect(LlamaRopeType.none.rawValue == -1, "LlamaRopeType.none should be -1")
        #expect(LlamaRopeType.norm.rawValue == 0, "LlamaRopeType.norm should be 0")
        #expect(LlamaRopeType.neox.rawValue == 2, "LlamaRopeType.neox should be 2")
        
        #expect(LlamaTokenType.undefined.rawValue == 0, "LlamaTokenType.undefined should be 0")
        #expect(LlamaTokenType.normal.rawValue == 1, "LlamaTokenType.normal should be 1")
        #expect(LlamaTokenType.unknown.rawValue == 2, "LlamaTokenType.unknown should be 2")
        
        #expect(LlamaTokenAttribute.undefined.rawValue == 0, "LlamaTokenAttribute.undefined should be 0")
        #expect(LlamaTokenAttribute.unknown.rawValue == 1, "LlamaTokenAttribute.unknown should be 1")
        #expect(LlamaTokenAttribute.normal.rawValue == 4, "LlamaTokenAttribute.normal should be 4")
        
        #expect(LlamaFileType.allF32.rawValue == 0, "LlamaFileType.allF32 should be 0")
        #expect(LlamaFileType.mostlyF16.rawValue == 1, "LlamaFileType.mostlyF16 should be 1")
        #expect(LlamaFileType.mostlyQ4_0.rawValue == 2, "LlamaFileType.mostlyQ4_0 should be 2")
    }
} 