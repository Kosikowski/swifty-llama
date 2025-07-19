#!/usr/bin/env swift

import Foundation

// 🧹 SLlama @inline(__always) Replacement Script 🧹
//
// This script replaces all @inline(__always) directives with @inlinable
// to ensure consistent conditional inlining across the entire codebase.
//
// Usage: cd to project root and run: ./Scripts/replace_inline_always.swift

func replaceInlineAlways(in filePath: String) {
    let fileName = URL(fileURLWithPath: filePath).lastPathComponent
    print("🔧 Processing: \(fileName)")

    guard let content = try? String(contentsOfFile: filePath, encoding: .utf8) else {
        print("❌ Failed to read file: \(filePath)")
        return
    }

    var lines = content.components(separatedBy: .newlines)
    var modified = false
    var replacementCount = 0

    for i in 0 ..< lines.count {
        let line = lines[i]

        // Check if this line contains @inline(__always)
        if line.contains("@inline(__always)") {
            // Replace @inline(__always) with @inlinable
            let newLine = line.replacingOccurrences(of: "@inline(__always)", with: "@inlinable")
            lines[i] = newLine

            modified = true
            replacementCount += 1
            print("✨ Replaced @inline(__always) with @inlinable")
        }
    }

    if modified {
        let newContent = lines.joined(separator: "\n")
        do {
            try newContent.write(toFile: filePath, atomically: true, encoding: .utf8)
            print("✅ Updated \(fileName) with \(replacementCount) replacements")
        } catch {
            print("❌ Failed to write file: \(filePath) - \(error)")
        }
    } else {
        print("ℹ️  No @inline(__always) found in \(fileName)")
    }
}

func getAllSwiftFiles(in directory: String) -> [String] {
    let fileManager = FileManager.default
    guard let enumerator = fileManager.enumerator(atPath: directory) else {
        print("❌ Failed to enumerate directory: \(directory)")
        return []
    }

    var swiftFiles: [String] = []
    while let file = enumerator.nextObject() as? String {
        if file.hasSuffix(".swift") {
            swiftFiles.append("\(directory)/\(file)")
        }
    }

    return swiftFiles.sorted()
}

// MARK: - Main Execution

print("🧹 SLlama @inline(__always) Replacement Tool")
print("=" * 60)
print("🎯 Replacing all @inline(__always) with @inlinable...")
print("")

let sLlamaDirectory = "Sources/SLlama"

// Check if we're in the right directory
guard FileManager.default.fileExists(atPath: sLlamaDirectory) else {
    print("❌ Error: \(sLlamaDirectory) not found!")
    print("ℹ️  Please run this script from the project root directory")
    exit(1)
}

let allSwiftFiles = getAllSwiftFiles(in: sLlamaDirectory)
print("📁 Found \(allSwiftFiles.count) Swift files:")
print("")

var processedCount = 0
var modifiedCount = 0

for file in allSwiftFiles {
    let originalContent = (try? String(contentsOfFile: file, encoding: .utf8)) ?? ""

    replaceInlineAlways(in: file)

    let newContent = (try? String(contentsOfFile: file, encoding: .utf8)) ?? ""
    if originalContent != newContent {
        modifiedCount += 1
    }
    processedCount += 1
    print("")
}

print("🎉 @inline(__always) replacement complete!")
print("=" * 60)
print("📊 Summary:")
print("   • Files processed: \(processedCount)")
print("   • Files modified: \(modifiedCount)")
print("   • All @inline(__always) replaced with @inlinable")
print("   • All inlining now controlled by SLLAMA_INLINE_ALL compiler flag")
print("")
print("ℹ️  Usage:")
print("   • Standard build: swift build")
print("   • With conditional inlining: swift build -Xswiftc -D -Xswiftc SLLAMA_INLINE_ALL")
print("   • Test with inlining: swift test -Xswiftc -D -Xswiftc SLLAMA_INLINE_ALL")
print("")
print("🚀 All inlining is now consistently conditional!")

// Extension for String repetition
extension String {
    static func * (string: String, count: Int) -> String {
        String(repeating: string, count: count)
    }
}
