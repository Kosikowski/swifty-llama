import Testing
import SwiftyLlamaCpp

@testable import SwiftyLlamaCpp

struct SLlamaTypesTests {
    
    @Test("Type aliases are properly defined")
    func testTypeAliases() throws {
        // Test that type aliases are properly defined
        #expect(type(of: SLlamaTokenNull) == SLlamaToken.self, "SLlamaTokenNull should be SLlamaToken type")
        #expect(type(of: SLlamaDefaultSeed) == UInt32.self, "SLlamaDefaultSeed should be UInt32 type")
        #expect(type(of: SLlamaFileMagicGGLA) == UInt32.self, "SLlamaFileMagicGGLA should be UInt32 type")
    }
    
    @Test("Enum type aliases work correctly")
    func testEnumTypeAliases() throws {
        // Test enum type aliases exist and have correct types
        #expect(SLlamaVocabType.none.rawValue == 0, "SLlamaVocabType.none should have raw value 0")
        #expect(SLlamaRopeType.norm.rawValue == 0, "SLlamaRopeType.norm should have raw value 0")
        #expect(SLlamaTokenType.normal.rawValue == 1, "SLlamaTokenType.normal should have raw value 1")
        #expect(SLlamaFileType.mostlyQ4_0.rawValue == 2, "SLlamaFileType.mostlyQ4_0 should have raw value 2")
    }
    
    @Test("Basic type definitions")
    func testBasicTypeDefinitions() throws {
        // Test that basic C types are properly bridged
        #expect(type(of: SLlamaTokenNull) == SLlamaToken.self, "SLlamaTokenNull should be SLlamaToken")
        #expect(type(of: SLlamaVocabType.none) == SLlamaVocabType.self, "SLlamaVocabType.none should be SLlamaVocabType")
        #expect(type(of: SLlamaTokenAttribute.undefined) == SLlamaTokenAttribute.self, "SLlamaTokenAttribute.undefined should be SLlamaTokenAttribute")
    }
    
    @Test("Pointer type aliases")
    func testPointerTypeAliases() throws {
        // Test that pointer type aliases are properly defined
        #expect(SLlamaModelPointer.self == OpaquePointer.self, "SLlamaModelPointer should be OpaquePointer")
        #expect(SLlamaContextPointer.self == OpaquePointer.self, "SLlamaContextPointer should be OpaquePointer")
        #expect(SLlamaVocabPointer.self == OpaquePointer.self, "SLlamaVocabPointer should be OpaquePointer")
    }
    
    @Test("Constants and magic values")
    func testConstantsAndMagicValues() throws {
        // Test that constants are properly defined
        #expect(SLlamaTokenNull == -1, "SLlamaTokenNull should be -1")
        #expect(SLlamaDefaultSeed == 0xFFFFFFFF, "SLlamaDefaultSeed should be 0xFFFFFFFF")
        #expect(SLlamaFileMagicGGLA == 0x67676C61, "SLlamaFileMagicGGLA should be correct magic")
        #expect(SLlamaFileMagicGGSN == 0x6767736E, "SLlamaFileMagicGGSN should be correct magic")
        #expect(SLlamaFileMagicGGSQ == 0x67677371, "SLlamaFileMagicGGSQ should be correct magic")
        #expect(SLlamaSessionMagic == 0x6767736E, "SLlamaSessionMagic should be correct magic")
        #expect(SLlamaSessionVersion == 9, "SLlamaSessionVersion should be 9")
        #expect(SLlamaStateSeqMagic == 0x67677371, "SLlamaStateSeqMagic should be correct magic")
        #expect(SLlamaStateSeqVersion == 2, "SLlamaStateSeqVersion should be 2")
    }
    
    @Test("Enum extensions")
    func testEnumExtensions() throws {
        // Test that enum extensions provide correct values
        #expect(SLlamaVocabType.none.rawValue == 0, "SLlamaVocabType.none should be 0")
        #expect(SLlamaVocabType.spm.rawValue == 1, "SLlamaVocabType.spm should be 1")
        #expect(SLlamaVocabType.bpe.rawValue == 2, "SLlamaVocabType.bpe should be 2")
        
        #expect(SLlamaRopeType.none.rawValue == -1, "SLlamaRopeType.none should be -1")
        #expect(SLlamaRopeType.norm.rawValue == 0, "SLlamaRopeType.norm should be 0")
        #expect(SLlamaRopeType.neox.rawValue == 2, "SLlamaRopeType.neox should be 2")
        
        #expect(SLlamaTokenType.undefined.rawValue == 0, "SLlamaTokenType.undefined should be 0")
        #expect(SLlamaTokenType.normal.rawValue == 1, "SLlamaTokenType.normal should be 1")
        #expect(SLlamaTokenType.unknown.rawValue == 2, "SLlamaTokenType.unknown should be 2")
        
        #expect(SLlamaTokenAttribute.undefined.rawValue == 0, "SLlamaTokenAttribute.undefined should be 0")
        #expect(SLlamaTokenAttribute.unknown.rawValue == 1, "SLlamaTokenAttribute.unknown should be 1")
        #expect(SLlamaTokenAttribute.normal.rawValue == 4, "SLlamaTokenAttribute.normal should be 4")
        
        #expect(SLlamaFileType.allF32.rawValue == 0, "SLlamaFileType.allF32 should be 0")
        #expect(SLlamaFileType.mostlyF16.rawValue == 1, "SLlamaFileType.mostlyF16 should be 1")
        #expect(SLlamaFileType.mostlyQ4_0.rawValue == 2, "SLlamaFileType.mostlyQ4_0 should be 2")
    }
} 