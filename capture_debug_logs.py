#!/usr/bin/env python3
"""
Capture debug logs with a specific prefix from Python output or log files.
Based on the analyze_position_logs.sh pattern from the app repo.
"""

import sys
import re
import subprocess
from typing import List, Optional
from pathlib import Path


def capture_debug_logs(prefix: str, log_sources: Optional[List[str]] = None, max_lines: int = 100) -> str:
    """
    Capture debug logs with a specific prefix pattern.
    
    Args:
        prefix: The debug prefix to search for (e.g., "ANIM_HANDOFF")
        log_sources: List of log files or commands to search (optional)
        max_lines: Maximum number of lines to return per category
    
    Returns:
        Formatted string with captured debug logs
    """
    if not log_sources:
        # Default: try to find recent Python output or test logs
        log_sources = [
            "*.log",
            "test_*.txt",
            "debug_*.txt",
            "output.txt"
        ]
    
    pattern = f"\\[{prefix}\\]"
    all_logs = []
    
    # Collect logs from files
    for source in log_sources:
        if "*" in source:
            # Glob pattern
            for file_path in Path(".").glob(source):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            matches = re.findall(f".*{pattern}.*", content, re.MULTILINE)
                            all_logs.extend(matches[-max_lines:])  # Keep only last N lines
                    except:
                        continue
        else:
            # Specific file
            if Path(source).exists():
                try:
                    with open(source, 'r') as f:
                        content = f.read()
                        matches = re.findall(f".*{pattern}.*", content, re.MULTILINE)
                        all_logs.extend(matches[-max_lines:])
                except:
                    continue
    
    # Also try to capture from recent Python test runs if they exist
    try:
        # Check if there's a recent test output
        result = subprocess.run(
            f"grep '{pattern}' test_*.py 2>/dev/null | head -{max_lines}",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            all_logs.extend(result.stdout.strip().split('\n'))
    except:
        pass
    
    # Also search for print statements in Python source files
    try:
        # Search in utils/animations for print statements with the prefix
        result = subprocess.run(
            f"grep -r 'print.*{pattern}' utils/animations/*.py 2>/dev/null | head -{max_lines}",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            # Extract just the print statement content
            for line in result.stdout.strip().split('\n'):
                if line:
                    # Extract the print statement part after the file path
                    if ':' in line:
                        _, print_content = line.split(':', 1)
                        # Clean up the print statement to show what would be logged
                        if 'print(' in print_content:
                            # Extract the string from the print statement
                            match = re.search(r'print\([^)]*["\']([^"\']*\[' + prefix + r'\][^"\']*)["\']', print_content)
                            if match:
                                all_logs.append(f"[SOURCE] {match.group(1)}")
    except:
        pass
    
    if not all_logs:
        return f"No debug logs found with prefix [{prefix}]"
    
    # Organize logs by category
    categories = {
        "Initialization": [],
        "Position/Layout": [],
        "Animation": [],
        "Handoff": [],
        "Error": [],
        "Other": []
    }
    
    for log in all_logs:
        log_lower = log.lower()
        if any(word in log_lower for word in ["init", "create", "setup", "start"]):
            categories["Initialization"].append(log)
        elif any(word in log_lower for word in ["position", "x=", "y=", "origin", "center"]):
            categories["Position/Layout"].append(log)
        elif any(word in log_lower for word in ["anim", "dissolve", "shrink", "phase"]):
            categories["Animation"].append(log)
        elif any(word in log_lower for word in ["handoff", "transfer", "frozen", "final"]):
            categories["Handoff"].append(log)
        elif any(word in log_lower for word in ["error", "fail", "wrong", "mismatch"]):
            categories["Error"].append(log)
        else:
            categories["Other"].append(log)
    
    # Format output
    output = []
    output.append(f"Debug Logs Capture [{prefix}]")
    output.append("=" * 50)
    output.append(f"Total logs found: {len(all_logs)}")
    output.append("")
    
    for category, logs in categories.items():
        if logs:
            output.append(f"{category} ({len(logs)} logs):")
            output.append("-" * 30)
            # Show last 10 logs per category
            for log in logs[-10:]:
                output.append(log)
            output.append("")
    
    return "\n".join(output)


def main():
    if len(sys.argv) < 2:
        print("Usage: python capture_debug_logs.py <PREFIX> [log_file1] [log_file2] ...")
        print("Example: python capture_debug_logs.py ANIM_HANDOFF")
        sys.exit(1)
    
    prefix = sys.argv[1]
    log_sources = sys.argv[2:] if len(sys.argv) > 2 else None
    
    result = capture_debug_logs(prefix, log_sources)
    print(result)


if __name__ == "__main__":
    main()