#!/usr/bin/env swift

import Foundation

// 🧹 SLlama Inlining Cleanup Script 🧹
//
// This script removes existing @inline(__always) directives to prepare for
// conditional inlining replacement.
//
// Usage: cd to project root and run: ./Scripts/cleanup_existing_inlining.swift

func cleanupFile(at path: String) {
    let fileName = URL(fileURLWithPath: path).lastPathComponent
    print("🧹 Cleaning up: \(fileName)")

    guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
        print("❌ Failed to read file: \(path)")
        return
    }

    var lines = content.components(separatedBy: .newlines)
    var modifiedLines: [String] = []
    var modified = false

    var i = 0
    while i < lines.count {
        let line = lines[i]

        // Skip @inline(__always) lines
        if line.trimmingCharacters(in: .whitespaces).hasPrefix("@inline(__always)") {
            print("🗑️  Removing @inline(__always) from: \(fileName)")
            modified = true
            i += 1
            continue
        }

        modifiedLines.append(line)
        i += 1
    }

    if modified {
        let newContent = modifiedLines.joined(separator: "\n")
        do {
            try newContent.write(toFile: path, atomically: true, encoding: .utf8)
            print("✅ Successfully cleaned: \(fileName)")
        } catch {
            print("❌ Failed to write file: \(path) - \(error)")
        }
    } else {
        print("ℹ️  No cleanup needed for: \(fileName)")
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

print("🧹 SLlama Inlining Cleanup Tool")
print("=" * 50)
print("🎯 Removing existing @inline(__always) directives...")
print("")

let sLlamaDirectory = "Sources/SLlama"

// Check if we're in the right directory
guard FileManager.default.fileExists(atPath: sLlamaDirectory) else {
    print("❌ Error: \(sLlamaDirectory) not found!")
    print("ℹ️  Please run this script from the project root directory")
    exit(1)
}

let allSwiftFiles = getAllSwiftFiles(in: sLlamaDirectory)
print("📁 Found \(allSwiftFiles.count) Swift files to check")
print("")

var processedCount = 0
var modifiedCount = 0

for file in allSwiftFiles {
    let fileName = URL(fileURLWithPath: file).lastPathComponent
    let originalContent = (try? String(contentsOfFile: file, encoding: .utf8)) ?? ""

    cleanupFile(at: file)

    let newContent = (try? String(contentsOfFile: file, encoding: .utf8)) ?? ""
    if originalContent != newContent {
        modifiedCount += 1
    }
    processedCount += 1
    print("")
}

print("🎉 Cleanup complete!")
print("=" * 50)
print("📊 Summary:")
print("   • Files processed: \(processedCount)")
print("   • Files modified: \(modifiedCount)")
print("")
print("ℹ️  Next step: Run ./Scripts/apply_comprehensive_inlining.swift")
print("   to apply conditional inlining to all files")

// Extension for String repetition
extension String {
    static func * (string: String, count: Int) -> String {
        String(repeating: string, count: count)
    }
}
