#!/usr/bin/env python3
"""
Create XML context from specified files
This script is deterministic - it just formats the given files into XML
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import subprocess
import re
from capture_debug_logs import capture_debug_logs


def get_project_tree(root_dir: str, relevant_files: list = None) -> str:
    """Generate a simplified folder tree showing only folders and relevant files"""
    
    tree_lines = []
    root_path = Path(root_dir)
    
    # Convert relevant files to set of Path objects for faster lookup
    relevant_paths = set()
    if relevant_files:
        for f in relevant_files:
            path = Path(f)
            relevant_paths.add(path)
            # Also add all parent directories
            for parent in path.parents:
                if parent != Path('.'):
                    relevant_paths.add(parent)
    
    def should_skip(item: Path) -> bool:
        """Check if item should be skipped"""
        name = item.name
        # Skip hidden files/folders (starting with .)
        if name.startswith('.'):
            return True
        # Skip common non-project directories
        if name in {'node_modules', '__pycache__', 'dist', 'build', 'coverage', 
                   'venv', '.venv', 'env', '.env', 'tmp', 'temp', 'cache',
                   '.pytest_cache', '.mypy_cache', 'htmlcov', '.tox',
                   'egg-info', '.egg-info', 'wheels', '.DS_Store'}:
            return True
        return False
    
    def add_tree(directory: Path, prefix: str = "", is_last: bool = True, depth: int = 0):
        """Recursively build tree structure"""
        if depth > 5:  # Limit depth to keep it manageable
            return
            
        dir_name = directory.name if directory != root_path else '.'
        
        # Add current directory
        if depth > 0:  # Don't show connector for root
            connector = "└── " if is_last else "├── "
            tree_lines.append(f"{prefix}{connector}{dir_name}/")
        else:
            tree_lines.append(f"{dir_name}/")
        
        # Prepare prefix for children
        if depth > 0:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension
        else:
            new_prefix = ""
        
        try:
            # Get items
            items = list(directory.iterdir())
            
            # Filter out unwanted items
            items = [item for item in items if not should_skip(item)]
            
            # Separate directories and files
            dirs = sorted([item for item in items if item.is_dir()], key=lambda x: x.name.lower())
            files = sorted([item for item in items if item.is_file()], key=lambda x: x.name.lower())
            
            # For files, only show if they're in the relevant_files list
            if relevant_files:
                rel_dir_path = directory.relative_to(root_path) if directory != root_path else Path('.')
                files = [f for f in files if str(rel_dir_path / f.name) in relevant_files or str(f.name) in relevant_files]
            else:
                files = []  # Don't show any files if no relevant files specified
            
            all_items = dirs + files
            
            # Add each item
            for i, item in enumerate(all_items):
                is_last_item = (i == len(all_items) - 1)
                
                if item.is_dir():
                    # Only recurse into directories that contain relevant files
                    if relevant_files:
                        rel_path = item.relative_to(root_path)
                        # Check if any relevant file is in this directory
                        has_relevant = any(str(rf).startswith(str(rel_path) + '/') for rf in relevant_files)
                        if has_relevant or str(rel_path) in [str(Path(rf).parent) for rf in relevant_files]:
                            add_tree(item, new_prefix, is_last_item, depth + 1)
                        else:
                            # Just show the directory name without recursing
                            connector = "└── " if is_last_item else "├── "
                            tree_lines.append(f"{new_prefix}{connector}{item.name}/")
                    else:
                        add_tree(item, new_prefix, is_last_item, depth + 1)
                else:
                    # File
                    connector = "└── " if is_last_item else "├── "
                    # Highlight relevant files
                    tree_lines.append(f"{new_prefix}{connector}{item.name} *")
                    
        except PermissionError:
            pass
    
    add_tree(root_path)
    
    return "\n".join(tree_lines)


def create_xml_from_files(issue: str, file_paths: list, project_root: str = ".", debug_prefix: str = None) -> str:
    """Create XML string from the given files"""
    
    # Create root element
    root = ET.Element("context")
    root.set("issue", issue)
    
    # Add files section
    files_elem = ET.SubElement(root, "files")
    files_elem.set("count", str(len(file_paths)))
    
    for file_path in file_paths:
        file_elem = ET.SubElement(files_elem, "file")
        file_elem.set("path", file_path)
        
        full_path = os.path.join(project_root, file_path)
        
        if not os.path.exists(full_path):
            file_elem.text = "[File not found]"
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # XML escape special characters
                content = content.replace('&', '&amp;')
                content = content.replace('<', '&lt;')
                content = content.replace('>', '&gt;')
                content = content.replace('"', '&quot;')
                content = content.replace("'", '&apos;')
                
                # Remove control characters except newline, return, tab
                content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                
                file_elem.text = content
        except UnicodeDecodeError:
            try:
                with open(full_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    content = content.replace('&', '&amp;')
                    content = content.replace('<', '&lt;')
                    content = content.replace('>', '&gt;')
                    content = content.replace('"', '&quot;')
                    content = content.replace("'", '&apos;')
                    content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                    file_elem.text = content
            except:
                file_elem.text = "[Binary file or encoding error]"
        except Exception as e:
            file_elem.text = f"[Error reading file: {str(e)}]"
    
    # Add project tree (showing only relevant files)
    tree_elem = ET.SubElement(root, "project_tree")
    tree_elem.text = get_project_tree(project_root, file_paths)
    
    # Add summary
    summary_elem = ET.SubElement(root, "summary")
    summary_elem.set("total_files", str(len(file_paths)))
    summary_elem.set("issue", issue)
    
    # Add debugging instructions
    if debug_prefix:
        debug_elem = ET.SubElement(root, "debugging")
        debug_text = f"""When fixing this issue, add debug log prints to help diagnose if the fix doesn't work.
All debug prints must follow the structure: [{debug_prefix}] message
Example: [{debug_prefix}] Letter positions frozen at: [(350, 180), (375, 180), ...]
This will help track the fix progress and identify any remaining issues."""
        debug_elem.text = debug_text
        
        # Capture existing debug logs if any
        try:
            captured_logs = capture_debug_logs(debug_prefix)
            if not captured_logs.startswith("No debug logs found"):
                debug_prints_elem = ET.SubElement(root, "debug_prints")
                debug_prints_elem.text = captured_logs
        except Exception as e:
            # If capture fails, just continue without debug logs
            pass
    
    # Convert to pretty XML string
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_filename_from_issue(issue: str) -> str:
    """Create a concise filename from issue description"""
    # Extract meaningful words
    words = re.findall(r'\b[a-zA-Z]+\b', issue)
    meaningful = [w.lower() for w in words if len(w) > 2][:3]
    
    if meaningful:
        return f"issue_description_{'_'.join(meaningful)}.xml"
    else:
        return "issue_description.xml"


def main():
    parser = argparse.ArgumentParser(description='Create XML context from specified files')
    parser.add_argument('--issue', required=True, help='Description of the issue')
    parser.add_argument('--files', nargs='+', required=True, help='List of file paths')
    parser.add_argument('--prefix', help='Debug prefix for log capture (e.g., ANIM_HANDOFF)')
    parser.add_argument('--output', help='Output filename (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    print(f"Creating XML for issue: {args.issue}")
    if args.prefix:
        print(f"Debug prefix: {args.prefix}")
    print(f"Processing {len(args.files)} files...")
    
    # Create XML
    xml_content = create_xml_from_files(args.issue, args.files, debug_prefix=args.prefix)
    
    # Determine output filename
    if args.output:
        filename = args.output
    else:
        filename = create_filename_from_issue(args.issue)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    print(f"XML saved to: {filename}")
    
    # Copy to clipboard (macOS)
    try:
        # Method 1: Try using pbcopy directly
        process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
        process.communicate(xml_content.encode('utf-8'))
        print("✓ Copied to clipboard using pbcopy")
    except (FileNotFoundError, OSError):
        try:
            # Method 2: Try using osascript as fallback
            escaped_content = xml_content.replace('\\', '\\\\').replace('"', '\\"')
            applescript = f'set the clipboard to "{escaped_content}"'
            subprocess.run(['osascript', '-e', applescript], check=True)
            print("✓ Copied to clipboard using osascript")
        except:
            # Method 3: Last resort - write to temp file and copy
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp:
                    tmp.write(xml_content)
                    tmp_path = tmp.name
                subprocess.run(f'cat {tmp_path} | pbcopy', shell=True, check=True)
                os.unlink(tmp_path)
                print("✓ Copied to clipboard via temp file")
            except:
                print(f"⚠️  Could not copy to clipboard automatically")
                print(f"   To copy manually, run: cat {filename} | pbcopy")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Files included: {len(args.files)}")
    print(f"  Output size: {len(xml_content) / 1024:.1f} KB")


if __name__ == "__main__":
    main()