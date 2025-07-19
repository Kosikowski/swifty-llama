# Pre-commit Setup Guide

This project uses pre-commit hooks to ensure code quality and consistency. The setup includes SwiftFormat, code quality checks, and other useful validations.

## What's Included

### 🎨 **Code Formatting**
- **SwiftFormat**: Automatically formats Swift code according to project standards
- **Configuration**: `.swiftformat` file with comprehensive formatting rules

### 🔍 **Code Quality Checks**
- **SwiftLint**: Optional Swift linting (if installed)
- **Build Check**: Ensures Swift code compiles before commit
- **Debug Print Check**: Prevents `print()` statements in `Sources/`
- **TODO/FIXME Check**: Warns about TODO/FIXME comments in production code

### 📁 **File Quality Checks**
- **Trailing Whitespace**: Removes trailing spaces
- **End of File**: Ensures files end with newline
- **Large Files**: Prevents accidentally committing large files (>1MB)
- **Merge Conflicts**: Detects unresolved merge conflict markers
- **Case Conflicts**: Prevents case-sensitive filename conflicts

### 📦 **Project Validation**
- **Package.swift**: Validates Swift package configuration
- **YAML/JSON/TOML**: Validates configuration file syntax
- **README**: Ensures README.md exists

## Usage

### Automatic (Recommended)
Pre-commit hooks run automatically on every `git commit`:

```bash
git add .
git commit -m "Your commit message"
# Hooks will run automatically and fix issues
```

### Manual Execution
Run hooks manually on all files:

```bash
python3 -m pre_commit run --all-files
```

Run specific hooks:

```bash
python3 -m pre_commit run swiftformat
python3 -m pre_commit run swift-build-check
```

### Skip Hooks (Emergency)
Skip all hooks for emergency commits:

```bash
git commit --no-verify -m "Emergency fix"
```

## Installation

The hooks are already installed in this repository. For new repositories:

1. Install pre-commit:
   ```bash
   python3 -m pip install pre-commit
   ```

2. Install hooks:
   ```bash
   python3 -m pre_commit install
   ```

## Configuration Files

### `.pre-commit-config.yaml`
Main pre-commit configuration with all hooks and settings.

### `.swiftformat`
SwiftFormat configuration with project-specific formatting rules:
- 4-space indentation
- 120 character line width
- Comprehensive formatting rules
- Excludes test utilities and C code

## Troubleshooting

### SwiftFormat Issues
If SwiftFormat fails:
1. Check if SwiftFormat is installed: `swiftformat --version`
2. Install if needed: `brew install swiftformat`
3. Check `.swiftformat` configuration

### Build Check Failures
If Swift build check fails:
1. Run `swift build` manually to see errors
2. Fix compilation errors
3. Try commit again

### Permission Issues
If hooks don't run:
1. Check hook installation: `ls -la .git/hooks/`
2. Reinstall hooks: `python3 -m pre_commit install`

### Update Hooks
Update to latest hook versions:

```bash
python3 -m pre_commit autoupdate
```

## Customization

### Disable Specific Hooks
Edit `.pre-commit-config.yaml` and comment out unwanted hooks.

### Modify SwiftFormat Rules
Edit `.swiftformat` to adjust formatting preferences.

### Add New Hooks
Add new hooks to `.pre-commit-config.yaml` following the existing pattern.

## Benefits

✅ **Consistent Code Style**: All code follows the same formatting standards  
✅ **Early Error Detection**: Catch issues before they reach CI/CD  
✅ **Automated Fixes**: Many issues are fixed automatically  
✅ **Team Productivity**: Reduces code review time spent on style issues  
✅ **Quality Assurance**: Prevents common mistakes and bad practices  

## Example Output

```
SwiftFormat..............................................................Passed
SwiftLint................................................(skipped: not installed)
Swift Build Check........................................................Passed
Check Debug Prints.......................................................Passed
Check TODOs/FIXMEs.......................................................Passed
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...............................................................Passed
✅ All checks passed!
``` 
