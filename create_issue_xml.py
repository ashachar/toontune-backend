#!/usr/bin/env python3
"""
Create an XML context document for issue debugging.
This script is 100% deterministic - it only formats what Claude provides.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def get_folder_tree(root_path, indent=""):
    """Get folder structure (directories only, no files)."""
    tree_lines = []
    items = sorted([d for d in os.listdir(root_path) 
                   if os.path.isdir(os.path.join(root_path, d)) 
                   and not d.startswith('.')])
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current = "└── " if is_last else "├── "
        tree_lines.append(f"{indent}{current}{item}/")
        
        # Skip deep nested folders
        if indent.count("│") < 3:
            next_indent = indent + ("    " if is_last else "│   ")
            subtree = get_folder_tree(os.path.join(root_path, item), next_indent)
            tree_lines.extend(subtree)
    
    return tree_lines

def read_file_content(filepath):
    """Read file content safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def main():
    parser = argparse.ArgumentParser(description='Create XML context for issue debugging')
    parser.add_argument('--issue', required=True, help='Complete issue description')
    parser.add_argument('--prefix', required=True, help='Debug prefix for logging (e.g., OPACITY_BLINK)')
    parser.add_argument('--files', nargs='+', required=True, help='List of relevant files')
    
    args = parser.parse_args()
    
    # Get project root (backend folder)
    project_root = Path(__file__).parent
    
    # Generate folder tree
    print("Generating folder structure...")
    tree_lines = get_folder_tree(project_root)
    folder_tree = "\n".join(tree_lines)
    
    # Create XML content
    xml_content = f"""<issue_context>
<debugging>
When fixing this issue, add debug log prints to help diagnose if the fix doesn't work.
All debug prints must follow the structure: [{args.prefix}] message
Example: [{args.prefix}] Alpha transition: motion_final={{motion_alpha:.3f}} -> dissolve_initial={{dissolve_alpha:.3f}}
</debugging>

<issue_description>
{args.issue}
</issue_description>

<project_structure>
backend/
{folder_tree}
</project_structure>

<relevant_files>
"""
    
    # Add file contents
    for filepath in args.files:
        file_path = Path(filepath)
        if file_path.exists():
            print(f"Reading {file_path.name}...")
            content = read_file_content(filepath)
            relative_path = file_path.relative_to(project_root) if project_root in file_path.parents else file_path
            xml_content += f"""
<file>
<path>{relative_path}</path>
<content>
{content}
</content>
</file>
"""
    
    xml_content += """
</relevant_files>
</issue_context>"""
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"issue_description_opacity_blink_{timestamp}.xml"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    print(f"\n✅ XML context saved to: {output_file}")
    
    # Copy to clipboard (macOS)
    try:
        subprocess.run(['pbcopy'], input=xml_content.encode('utf-8'), check=True)
        print("📋 Copied to clipboard!")
    except:
        print("⚠️  Could not copy to clipboard")
    
    print(f"\nDebug prefix for this issue: [{args.prefix}]")

if __name__ == "__main__":
    main()