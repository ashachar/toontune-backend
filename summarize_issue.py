#!/usr/bin/env python3
"""
Summarize Issue - Collect relevant files and create XML context
Usage: python summarize_issue.py "issue description"
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import re
from typing import List, Set
import argparse


def get_project_tree(root_dir: str, ignore_dirs: Set[str] = None) -> str:
    """Generate a folder tree of the project (folders only, no files)"""
    if ignore_dirs is None:
        ignore_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv', 
            'dist', 'build', '.next', '.cache', 'coverage', '.pytest_cache',
            'output', 'outputs', '.ipynb_checkpoints', 'checkpoints',
            '.DS_Store', '.idea', '.vscode', '__MACOSX', '.mypy_cache',
            'egg-info', '.egg-info', '.tox', 'htmlcov', '.coverage',
            'site-packages', 'lib', 'bin', 'include', 'share'
        }
    
    tree_lines = []
    
    def add_tree(directory: Path, prefix: str = "", is_last: bool = True):
        """Recursively build tree structure (folders only)"""
        dir_name = directory.name or '.'
        
        # Add current directory
        connector = "└── " if is_last else "├── "
        tree_lines.append(f"{prefix}{connector}{dir_name}/")
        
        # Prepare prefix for children
        extension = "    " if is_last else "│   "
        new_prefix = prefix + extension
        
        try:
            # Get only subdirectories, not files
            subdirs = []
            for item in directory.iterdir():
                if item.is_dir() and item.name not in ignore_dirs and not item.name.startswith('.'):
                    subdirs.append(item)
            
            # Sort directories
            subdirs = sorted(subdirs, key=lambda x: x.name.lower())
            
            # Add each subdirectory
            for i, subdir in enumerate(subdirs):
                is_last_item = (i == len(subdirs) - 1)
                add_tree(subdir, new_prefix, is_last_item)
                
        except PermissionError:
            pass
    
    # Start from root
    root_path = Path(root_dir)
    add_tree(root_path, "", True)
    
    return "\n".join(tree_lines)


def find_relevant_files(issue: str, project_root: str) -> List[str]:
    """Find files relevant to the issue using various search methods"""
    relevant_files = set()
    
    # First, check if specific files are mentioned in the issue
    specific_files = extract_specific_files(issue)
    if specific_files:
        # If specific files are mentioned, focus ONLY on those and related files
        for file in specific_files:
            # Add the specific file
            if os.path.exists(os.path.join(project_root, file)):
                relevant_files.add(file)
            
            # Add files in the same directory
            file_dir = os.path.dirname(file)
            if file_dir and os.path.exists(os.path.join(project_root, file_dir)):
                for item in os.listdir(os.path.join(project_root, file_dir)):
                    item_path = os.path.join(file_dir, item)
                    if item.endswith(('.py', '.js', '.ts', '.tsx')):
                        relevant_files.add(item_path)
            
            # Look for test files that might use this file
            base_name = os.path.basename(file).replace('.py', '').replace('.js', '')
            test_patterns = [f'test_{base_name}', f'{base_name}_test', f'test_*{base_name}*']
            for root, dirs, files in os.walk(project_root):
                dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', 'venv', '.venv'}]
                for f in files:
                    for pattern in test_patterns:
                        if f.startswith('test_') and base_name in f.lower():
                            relevant_files.add(os.path.relpath(os.path.join(root, f), project_root))
        
        # If we found specific files, return just those
        if relevant_files:
            return sorted(list(relevant_files))
    
    # Otherwise, extract potential keywords from issue
    keywords = extract_keywords(issue)
    
    # Common CODE file extensions only (no images/assets)
    code_extensions = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
        '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.sh',
        '.yaml', '.yml', '.json', '.xml', '.toml', '.md'
    }
    
    # Files to exclude even if they match keywords
    exclude_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico',
        '.mp4', '.mp3', '.wav', '.avi', '.mov', '.webm',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe',
        '.DS_Store', '.gitignore', '.dockerignore'
    }
    
    # Search strategies
    
    # 1. Direct filename matches (only code files)
    for root, dirs, files in os.walk(project_root):
        # Skip common ignore directories
        dirs[:] = [d for d in dirs if d not in {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            'dist', 'build', '.next', '.cache', 'coverage', '.pytest_cache',
            'sam2', 'cartoon-head-detection', 'ai-docs', 'docs'
        }]
        
        for file in files:
            file_lower = file.lower()
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, project_root)
            
            # Skip non-code files
            if any(file_lower.endswith(ext) for ext in exclude_extensions):
                continue
            
            # Only include Python and JS/TS files primarily
            primary_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx'}
            if not any(file_lower.endswith(ext) for ext in primary_extensions):
                # Skip non-primary code files unless they match a keyword strongly
                continue
            
            # Check if any keyword matches filename
            for keyword in keywords:
                if keyword.lower() in file_lower:
                    relevant_files.add(rel_path)
                    break
    
    # 2. Search file contents using grep (only primary code files)
    primary_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx'}
    for keyword in keywords:
        try:
            # Build include patterns for grep (only primary extensions)
            include_patterns = []
            for ext in primary_extensions:
                include_patterns.extend(['--include', f'*{ext}'])
            
            # Exclude directories from grep search
            exclude_dirs = ['--exclude-dir=.git', '--exclude-dir=node_modules', 
                          '--exclude-dir=venv', '--exclude-dir=.venv',
                          '--exclude-dir=sam2', '--exclude-dir=cartoon-head-detection',
                          '--exclude-dir=ai-docs', '--exclude-dir=docs']
            
            # Use grep to find files containing the keyword
            grep_cmd = ['grep', '-r', '-l', '-i'] + include_patterns + exclude_dirs + [keyword, project_root]
            result = subprocess.run(
                grep_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        rel_path = os.path.relpath(line, project_root)
                        # Double-check it's a code file and not excluded
                        if (any(rel_path.endswith(ext) for ext in primary_extensions) and
                            not any(rel_path.endswith(ext) for ext in exclude_extensions)):
                            relevant_files.add(rel_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # 3. Look for specific patterns in issue
    # Check for file paths mentioned in the issue
    path_pattern = r'[\w/\\]+\.\w+'
    potential_paths = re.findall(path_pattern, issue)
    for path in potential_paths:
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            relevant_files.add(path)
    
    # 4. Add only critical config files (not documentation)
    important_files = [
        'package.json', 'requirements.txt', 'pyproject.toml',
        'setup.py', 'setup.cfg'
    ]
    
    for imp_file in important_files:
        full_path = os.path.join(project_root, imp_file)
        if os.path.exists(full_path):
            relevant_files.add(imp_file)
    
    return sorted(list(relevant_files))


def extract_specific_files(issue: str) -> List[str]:
    """Extract specific file names or paths mentioned in the issue"""
    specific_files = []
    
    # Look for file paths with extensions
    file_pattern = r'[\w/\-_]+\.(?:py|js|jsx|ts|tsx|java|cpp|c|h|go|rs|rb|php)'
    matches = re.findall(file_pattern, issue)
    specific_files.extend(matches)
    
    # Look for module/class names that might be files
    # e.g., "text_behind_segment" -> "text_behind_segment.py"
    module_pattern = r'\b([a-z_]+(?:_[a-z]+)+)\b'
    potential_modules = re.findall(module_pattern, issue.lower())
    
    for module in potential_modules:
        # Check if it might be a Python file
        if '_' in module and len(module) > 5:
            # Search for this as a file
            for ext in ['.py', '.js', '.ts']:
                potential_path = f"{module}{ext}"
                specific_files.append(potential_path)
                specific_files.append(f"utils/{module}{ext}")
                specific_files.append(f"utils/animations/{module}{ext}")
    
    return list(set(specific_files))


def extract_keywords(issue: str) -> List[str]:
    """Extract meaningful keywords from issue description"""
    # Remove common words
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was',
        'were', 'to', 'of', 'for', 'with', 'in', 'by', 'from', 'about',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'that',
        'this', 'those', 'these', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
        'all', 'would', 'there', 'could', 'should', 'may', 'might', 'must',
        'can', 'will', 'do', 'does', 'did', 'have', 'has', 'had', 'make',
        'need', 'want', 'try', 'trying', 'tried', 'says', 'said', 'get',
        'got', 'getting', 'give', 'gave', 'given', 'take', 'took', 'taken'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z_]\w*\b', issue.lower())
    
    # Filter out stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    # Limit to most relevant keywords
    return unique_keywords[:10]


def create_xml_context(issue: str, files: List[str], project_root: str) -> ET.Element:
    """Create XML structure with file contents and project tree"""
    
    # Create root element
    root = ET.Element("context")
    root.set("issue", issue)
    
    # Add files section
    files_elem = ET.SubElement(root, "files")
    files_elem.set("count", str(len(files)))
    
    for file_path in files:
        file_elem = ET.SubElement(files_elem, "file")
        file_elem.set("path", file_path)
        
        full_path = os.path.join(project_root, file_path)
        
        try:
            # Try to read as text
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Escape special XML characters
                content = content.replace('&', '&amp;')
                content = content.replace('<', '&lt;')
                content = content.replace('>', '&gt;')
                content = content.replace('"', '&quot;')
                content = content.replace("'", '&apos;')
                # Remove any null bytes or control characters
                content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                file_elem.text = content
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(full_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    # Escape special XML characters
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
    
    # Add project tree
    tree_elem = ET.SubElement(root, "project_tree")
    tree_elem.text = get_project_tree(project_root)
    
    # Add summary
    summary_elem = ET.SubElement(root, "summary")
    summary_elem.set("total_files", str(len(files)))
    summary_elem.set("issue_keywords", ", ".join(extract_keywords(issue)))
    
    return root


def prettify_xml(elem: ET.Element) -> str:
    """Return a pretty-printed XML string"""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_concise_name(issue: str) -> str:
    """Create a concise filename from issue description"""
    # Extract first few meaningful words
    words = re.findall(r'\b[a-zA-Z]+\b', issue)
    
    # Filter short words
    meaningful = [w for w in words if len(w) > 2][:3]
    
    if meaningful:
        return "_".join(meaningful).lower()
    else:
        return "issue"


def main():
    parser = argparse.ArgumentParser(description='Summarize project files relevant to an issue')
    parser.add_argument('issue', help='Description of the issue')
    parser.add_argument('--root', default='.', help='Project root directory')
    parser.add_argument('--max-files', type=int, default=50, help='Maximum number of files to include')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = os.path.abspath(args.root)
    
    print(f"Analyzing issue: {args.issue}")
    print(f"Project root: {project_root}")
    
    # Find relevant files
    print("\nSearching for relevant files...")
    relevant_files = find_relevant_files(args.issue, project_root)
    
    # Limit number of files
    if len(relevant_files) > args.max_files:
        print(f"Found {len(relevant_files)} files, limiting to {args.max_files}")
        relevant_files = relevant_files[:args.max_files]
    else:
        print(f"Found {len(relevant_files)} relevant files")
    
    # Display found files
    print("\nRelevant files:")
    for f in relevant_files[:10]:  # Show first 10
        print(f"  - {f}")
    if len(relevant_files) > 10:
        print(f"  ... and {len(relevant_files) - 10} more")
    
    # Create XML context
    print("\nCreating XML context...")
    xml_root = create_xml_context(args.issue, relevant_files, project_root)
    
    # Convert to pretty XML string
    xml_string = prettify_xml(xml_root)
    
    # Create filename
    concise_name = create_concise_name(args.issue)
    filename = f"issue_description_{concise_name}.xml"
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_string)
    
    print(f"\nXML context saved to: {filename}")
    
    # Copy to clipboard
    try:
        # Try macOS pbcopy
        process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
        process.communicate(xml_string.encode('utf-8'))
        print("✓ Copied to clipboard")
    except FileNotFoundError:
        try:
            # Try Linux xclip
            process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
            process.communicate(xml_string.encode('utf-8'))
            print("✓ Copied to clipboard")
        except FileNotFoundError:
            print("ℹ Could not copy to clipboard (pbcopy/xclip not found)")
            print(f"Run: cat {filename} | pbcopy")
    
    # Show summary
    print(f"\nSummary:")
    print(f"  Issue: {args.issue}")
    print(f"  Files collected: {len(relevant_files)}")
    print(f"  Output: {filename}")
    print(f"  Size: {len(xml_string) / 1024:.1f} KB")


if __name__ == "__main__":
    main()