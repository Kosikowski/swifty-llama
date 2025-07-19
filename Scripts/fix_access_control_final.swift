#!/usr/bin/env swift

import Foundation

// 🔧 SLlama Final Access Control Fix Script 🔧
//
// This script fixes all access control issues for conditional inlining:
// 1. Removes duplicate @usableFromInline attributes
// 2. Changes private properties to internal @usableFromInline
// 3. Changes private methods to internal @usableFromInline
// 4. Removes @usableFromInline from public initializers
//
// Usage: cd to project root and run: ./Scripts/fix_access_control_final.swift

func fixAccessControl(in filePath: String) {
    let fileName = URL(fileURLWithPath: filePath).lastPathComponent
    print("🔧 Processing: \(fileName)")

    guard let content = try? String(contentsOfFile: filePath, encoding: .utf8) else {
        print("❌ Failed to read file: \(filePath)")
        return
    }

    var lines = content.components(separatedBy: .newlines)
    var modified = false
    var fixCount = 0

    // Fix 1: Remove duplicate @usableFromInline attributes
    var i = 0
    while i < lines.count - 1 {
        let currentLine = lines[i].trimmingCharacters(in: .whitespaces)
        let nextLine = lines[i + 1].trimmingCharacters(in: .whitespaces)

        if currentLine == "#if SLLAMA_INLINE_ALL", nextLine == "#if SLLAMA_INLINE_ALL" {
            // Found duplicate #if blocks, remove the second one and its @usableFromInline
            lines.remove(at: i + 1) // Remove duplicate #if
            if i + 1 < lines.count, lines[i + 1].trimmingCharacters(in: .whitespaces) == "@usableFromInline" {
                lines.remove(at: i + 1) // Remove duplicate @usableFromInline
            }
            if i + 1 < lines.count, lines[i + 1].trimmingCharacters(in: .whitespaces) == "#endif" {
                lines.remove(at: i + 1) // Remove duplicate #endif
            }
            modified = true
            fixCount += 1
            print("  🧹 Removed duplicate @usableFromInline")
        }
        i += 1
    }

    // Fix 2: Change private properties to internal @usableFromInline
    for i in 0 ..< lines.count {
        let line = lines[i]
        if line.contains("private var"), line.contains(":") {
            let newLine = line.replacingOccurrences(of: "private var", with: "internal var")
            if newLine != line {
                lines[i] = newLine
                modified = true
                fixCount += 1
                print("  🔄 Changed private var to internal var")
            }
        }
    }

    // Fix 3: Change private methods to internal @usableFromInline
    for i in 0 ..< lines.count {
        let line = lines[i]
        if line.contains("private func") {
            let newLine = line.replacingOccurrences(of: "private func", with: "internal func")
            if newLine != line {
                lines[i] = newLine
                modified = true
                fixCount += 1
                print("  🔄 Changed private func to internal func")
            }
        }
    }

    // Fix 4: Remove @usableFromInline from public initializers
    for i in 0 ..< lines.count {
        let line = lines[i]
        if line.contains("public init(") {
            // Look for @usableFromInline before this line
            var j = i - 1
            while j >= 0, lines[j].trimmingCharacters(in: .whitespaces).isEmpty {
                j -= 1
            }
            if j >= 0, lines[j].contains("@usableFromInline") {
                // Remove the @usableFromInline line and its #if/#endif
                if j > 0, lines[j - 1].contains("#if SLLAMA_INLINE_ALL") {
                    lines.remove(at: j - 1) // Remove #if
                }
                lines.remove(at: j - 1) // Remove @usableFromInline
                if j - 1 < lines.count, lines[j - 1].contains("#endif") {
                    lines.remove(at: j - 1) // Remove #endif
                }
                modified = true
                fixCount += 1
                print("  🧹 Removed @usableFromInline from public initializer")
            }
        }
    }

    if modified {
        let newContent = lines.joined(separator: "\n")
        try? newContent.write(toFile: filePath, atomically: true, encoding: .utf8)
        print("  ✅ Fixed \(fixCount) access control issues")
    } else {
        print("  ✅ No access control issues found")
    }
}

// Main execution
print("🔧 SLlama Final Access Control Fix Script")
print(String(repeating: "=", count: 50))

let sourceDir = "Sources/SLlama"
let fileManager = FileManager.default

guard let files = try? fileManager.contentsOfDirectory(atPath: sourceDir) else {
    print("❌ Failed to read source directory")
    exit(1)
}

let swiftFiles = files.filter { $0.hasSuffix(".swift") }.sorted()
var totalFixed = 0

for file in swiftFiles {
    let filePath = "\(sourceDir)/\(file)"
    fixAccessControl(in: filePath)
}

print(String(repeating: "=", count: 50))
print("🎉 Access control fix complete!")
