#!/usr/bin/env swift

import Foundation

// 🔧 SLlama Inlinable Access Control Fix Script 🔧
//
// This script adds @usableFromInline to private properties that are accessed
// by @inlinable methods to fix compilation errors.
//
// Usage: cd to project root and run: ./Scripts/fix_inlinable_access.swift

func fixInlinableAccess(in filePath: String) {
    let fileName = URL(fileURLWithPath: filePath).lastPathComponent
    print("🔧 Processing: \(fileName)")

    guard let content = try? String(contentsOfFile: filePath, encoding: .utf8) else {
        print("❌ Failed to read file: \(filePath)")
        return
    }

    var lines = content.components(separatedBy: .newlines)
    var modified = false
    var fixCount = 0

    for i in 0 ..< lines.count {
        let line = lines[i]

        // Look for private properties that might be accessed by inlinable methods
        if line.contains("private"),
           line.contains("context") ||
           line.contains("sampler") ||
           line.contains("chain") ||
           line.contains("vocab") ||
           line.contains("isMonitoring") ||
           line.contains("monitoringTimer") ||
           line.contains("metrics") ||
           line.contains("getCurrentCPUUsage")
        {
            // Check if this file has @inlinable methods
            if content.contains("@inlinable") {
                // Add @usableFromInline before private properties
                let indentation = getIndentation(of: line)
                let trimmedLine = line.trimmingCharacters(in: .whitespaces)

                if trimmedLine.hasPrefix("private") {
                    // Replace private with @usableFromInline private
                    let newLine = line.replacingOccurrences(
                        of: "private",
                        with: "#if SLLAMA_INLINE_ALL\n\(indentation)@usableFromInline\n#endif\n\(indentation)private"
                    )
                    lines[i] = newLine

                    modified = true
                    fixCount += 1
                    print("✨ Added @usableFromInline to: \(trimmedLine)")
                }
            }
        }
    }

    if modified {
        let newContent = lines.joined(separator: "\n")
        do {
            try newContent.write(toFile: filePath, atomically: true, encoding: .utf8)
            print("✅ Updated \(fileName) with \(fixCount) fixes")
        } catch {
            print("❌ Failed to write file: \(filePath) - \(error)")
        }
    } else {
        print("ℹ️  No access control fixes needed in \(fileName)")
    }
}

func getIndentation(of line: String) -> String {
    let leading = line.prefix { $0.isWhitespace }
    return String(leading)
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

print("🔧 SLlama Inlinable Access Control Fix Tool")
print("=" * 60)
print("🎯 Adding @usableFromInline to private properties accessed by @inlinable methods...")
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

    fixInlinableAccess(in: file)

    let newContent = (try? String(contentsOfFile: file, encoding: .utf8)) ?? ""
    if originalContent != newContent {
        modifiedCount += 1
    }
    processedCount += 1
    print("")
}

print("🎉 Access control fixes complete!")
print("=" * 60)
print("📊 Summary:")
print("   • Files processed: \(processedCount)")
print("   • Files modified: \(modifiedCount)")
print("   • Added @usableFromInline to private properties")
print("   • Fixed compilation errors for @inlinable methods")
print("")
print("ℹ️  Usage:")
print("   • Standard build: swift build")
print("   • With conditional inlining: swift build -Xswiftc -D -Xswiftc SLLAMA_INLINE_ALL")
print("   • Test with inlining: swift test -Xswiftc -D -Xswiftc SLLAMA_INLINE_ALL")
print("")
print("🚀 All @inlinable methods should now compile correctly!")

// Extension for String repetition
extension String {
    static func * (string: String, count: Int) -> String {
        String(repeating: string, count: count)
    }
}
