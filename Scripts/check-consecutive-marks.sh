#!/bin/bash

# Script to check for consecutive MARK comments in Swift files
# Usage: ./Scripts/check-consecutive-marks.sh

echo "🔍 Checking for consecutive MARK comments..."

found_issues=0

# Function to check a single file
check_file() {
    local file="$1"
    local prev_line=""
    local prev_line_num=0
    local line_num=0
    
    while IFS= read -r line; do
        ((line_num++))
        
        # Check if current line is a MARK comment
        if [[ "$line" =~ ^[[:space:]]*//[[:space:]]*MARK: ]]; then
            # Check if previous non-empty line was also a MARK comment
            if [[ "$prev_line" =~ ^[[:space:]]*//[[:space:]]*MARK: ]]; then
                echo "❌ Consecutive MARK comments found in $file:"
                echo "   Line $prev_line_num: $prev_line"
                echo "   Line $line_num: $line"
                echo ""
                found_issues=1
            fi
        fi
        
        # Update previous line only if current line is not empty or whitespace-only
        if [[ ! "$line" =~ ^[[:space:]]*$ ]]; then
            prev_line="$line"
            prev_line_num=$line_num
        fi
    done < "$file"
}

# Check all Swift files in Sources and Tests
while IFS= read -r -d '' file; do
    check_file "$file"
done < <(find Sources Tests -name "*.swift" -type f -print0 2>/dev/null)

if [ $found_issues -eq 0 ]; then
    echo "✅ No consecutive MARK comments found!"
else
    echo "💡 Suggestion: Combine related MARK comments or add content between them."
    echo ""
    echo "Examples:"
    echo "  Bad:  // MARK: Functions"
    echo "        // MARK: System Configuration"
    echo ""
    echo "  Good: // MARK: - System Configuration Functions"
    echo "  Or:   // MARK: Functions"
    echo "        "
    echo "        // Content here..."
    echo "        "
    echo "        // MARK: - System Configuration"
fi

exit $found_issues 