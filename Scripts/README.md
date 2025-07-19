# 🔧 SLlama Maintenance Scripts

This directory contains useful maintenance and optimization scripts for the SLlama project.

## 🚀 Available Scripts

### `apply_comprehensive_inlining.swift`

**Purpose**: Applies conditional inlining optimization to all SLlama source files for enhanced performance.

**Features**:
- **Two-tier optimization strategy**:
  - Core classes (SLlama, SLlamaCore): Conditional `@inlinable` (user-controlled)
  - All other classes: Aggressive `@inline(__always)` (always active)
- **Smart detection**: Skips files that already have inlining applied
- **Comprehensive coverage**: Optimizes 165+ public methods across 22 classes
- **Progress reporting**: Detailed output showing what's being processed

**Usage**:
```bash
# Run from project root
./Scripts/apply_comprehensive_inlining.swift
```

**Expected Impact**:
- 10-25% performance improvement for inference-heavy workloads
- Reduced function call overhead across entire API surface
- Enhanced compiler optimizations with cross-module inlining

**When to Use**:
- After adding new public methods to any SLlama class
- When setting up performance-critical builds
- For maintenance after major API changes

## 📋 Prerequisites

- Swift compiler available in PATH
- Run from the SLlama project root directory
- Ensure all files are committed before running (for easy rollback if needed)

## 🎯 Performance Notes

The inlining scripts are designed to:
- Maintain backwards compatibility
- Provide zero overhead when conditional features are disabled
- Enable maximum performance when enabled
- Keep the codebase maintainable with automated tooling

For more details about conditional inlining, see `CONDITIONAL_INLINING.md` in the project root. 