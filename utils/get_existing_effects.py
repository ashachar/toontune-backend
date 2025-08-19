#!/usr/bin/env python3
"""
Script to extract all animation and editing effect function signatures
and copy them to clipboard for easy reference.
"""

import ast
import inspect
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
import importlib
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_function_signatures_from_file(file_path: Path) -> List[Tuple[str, str]]:
    """
    Extract function names and signatures from a Python file using AST.
    
    Returns list of (function_name, signature) tuples.
    """
    signatures = []
    
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private/helper functions
                if node.name.startswith('_'):
                    continue
                
                # Build function signature
                args = []
                defaults_start = len(node.args.args) - len(node.args.defaults)
                
                for i, arg in enumerate(node.args.args):
                    arg_str = arg.arg
                    
                    # Add type annotation if present
                    if arg.annotation:
                        try:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        except:
                            pass
                    
                    # Add default value if present
                    if i >= defaults_start:
                        default_idx = i - defaults_start
                        try:
                            default_val = ast.unparse(node.args.defaults[default_idx])
                            arg_str += f" = {default_val}"
                        except:
                            pass
                    
                    args.append(arg_str)
                
                # Build return type if present
                return_type = ""
                if node.returns:
                    try:
                        return_type = f" -> {ast.unparse(node.returns)}"
                    except:
                        pass
                
                signature = f"{node.name}({', '.join(args)}){return_type}"
                signatures.append((node.name, signature))
                
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return signatures


def extract_class_methods_from_file(file_path: Path) -> List[Tuple[str, str]]:
    """
    Extract animation class names and their __init__ signatures from a file.
    
    Returns list of (class_name, init_signature) tuples.
    """
    signatures = []
    
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Skip private classes
                if node.name.startswith('_'):
                    continue
                
                # Find __init__ method
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        # Build init signature
                        args = []
                        defaults_start = len(item.args.args) - len(item.args.defaults)
                        
                        for i, arg in enumerate(item.args.args):
                            # Skip 'self'
                            if arg.arg == 'self':
                                continue
                            
                            arg_str = arg.arg
                            
                            # Add type annotation if present
                            if arg.annotation:
                                try:
                                    arg_str += f": {ast.unparse(arg.annotation)}"
                                except:
                                    pass
                            
                            # Add default value if present
                            if i >= defaults_start:
                                default_idx = i - defaults_start
                                if default_idx >= 0 and default_idx < len(item.args.defaults):
                                    try:
                                        default_val = ast.unparse(item.args.defaults[default_idx])
                                        arg_str += f" = {default_val}"
                                    except:
                                        pass
                            
                            args.append(arg_str)
                        
                        signature = f"{node.name}({', '.join(args)})"
                        signatures.append((node.name, signature))
                        break
                        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return signatures


def get_all_effects() -> str:
    """
    Collect all animation and editing effect signatures.
    
    Returns formatted string with all effects.
    """
    output = []
    output.append("=" * 80)
    output.append("TOONTUNE VIDEO EFFECTS AND ANIMATIONS REFERENCE")
    output.append("=" * 80)
    output.append("")
    
    # Get paths
    utils_dir = Path(__file__).parent
    editing_tricks_dir = utils_dir / "editing_tricks"
    animations_dir = utils_dir / "animations"
    
    # ========== EDITING TRICKS ==========
    output.append("=" * 80)
    output.append("EDITING TRICKS (utils/editing_tricks/)")
    output.append("=" * 80)
    output.append("")
    
    # Process each editing tricks module
    modules_order = [
        "color_effects.py",
        "text_effects.py", 
        "motion_effects.py",
        "layout_effects.py"
    ]
    
    for module_file in modules_order:
        module_path = editing_tricks_dir / module_file
        if module_path.exists():
            module_name = module_file.replace('.py', '').replace('_', ' ').title()
            output.append(f"### {module_name}")
            output.append("-" * 40)
            
            signatures = extract_function_signatures_from_file(module_path)
            
            # Filter out helper functions
            main_functions = [
                (name, sig) for name, sig in signatures 
                if name.startswith('apply_') or name.startswith('add_')
            ]
            
            if main_functions:
                for name, signature in main_functions:
                    output.append(f"{signature}")
                output.append("")
            else:
                output.append("No public functions found.")
                output.append("")
    
    # ========== ANIMATIONS ==========
    output.append("=" * 80)
    output.append("ANIMATIONS (utils/animations/)")
    output.append("=" * 80)
    output.append("")
    output.append("All animation classes inherit from Animation base class.")
    output.append("Common parameters: element_path, background_path, position, duration, fps")
    output.append("")
    output.append("Size-transforming animations (inherit from ScaleTransformAnimation):")
    output.append("  - Support start_width, start_height, end_width, end_height parameters")
    output.append("  - Include: ZoomIn, ZoomOut, StretchSquash, DepthZoom")
    output.append("")
    
    # Get all animation files
    animation_files = sorted([
        f for f in animations_dir.glob("*.py")
        if f.name not in ["__init__.py", "animate.py", "__pycache__"]
    ])
    
    # Group animations by category
    categories = {
        "Motion Effects": ["bounce.py", "carousel.py", "depth_zoom.py", "emergence_from_static_point.py", 
                          "floating.py", "roll.py", "rotate_3d.py", "spin.py", "warp.py", "wave.py"],
        "Transition Effects": ["fade_in.py", "fade_out.py", "flip.py", "slide_in.py", "slide_out.py", 
                               "zoom_in.py", "zoom_out.py"],
        "Visual Effects": ["glitch.py", "lens_flare.py", "neon_glow.py", "particles.py", "shatter.py"],
        "Text Effects": ["split_text.py", "typewriter.py", "word_buildup.py"],
        "Transform Effects": ["skew.py", "stretch_squash.py"]
    }
    
    # Process by category
    for category, files in categories.items():
        category_animations = []
        
        for file_name in files:
            file_path = animations_dir / file_name
            if file_path.exists():
                class_sigs = extract_class_methods_from_file(file_path)
                category_animations.extend(class_sigs)
        
        if category_animations:
            output.append(f"### {category}")
            output.append("-" * 40)
            for name, signature in sorted(category_animations):
                output.append(f"{signature}")
            output.append("")
    
    # ========== SUMMARY ==========
    output.append("=" * 80)
    output.append("SUMMARY")
    output.append("=" * 80)
    
    # Count effects
    editing_count = 0
    for module_file in modules_order:
        module_path = editing_tricks_dir / module_file
        if module_path.exists():
            sigs = extract_function_signatures_from_file(module_path)
            editing_count += len([s for s in sigs if s[0].startswith(('apply_', 'add_'))])
    
    animation_count = 0
    for f in animation_files:
        sigs = extract_class_methods_from_file(f)
        animation_count += len(sigs)
    
    output.append(f"Total Editing Effects: {editing_count}")
    output.append(f"Total Animation Classes: {animation_count}")
    output.append(f"Grand Total: {editing_count + animation_count} effects")
    output.append("")
    output.append("=" * 80)
    output.append("END OF EFFECTS REFERENCE")
    output.append("=" * 80)
    
    return "\n".join(output)


def main():
    """Main function to generate effects list and copy to clipboard."""
    
    print("Extracting all video effects and animations...")
    
    # Get all effects
    effects_text = get_all_effects()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(effects_text)
        temp_file = f.name
    
    # Also save a permanent copy
    output_file = Path(__file__).parent / "all_effects_reference.txt"
    with open(output_file, 'w') as f:
        f.write(effects_text)
    
    # Copy to clipboard
    try:
        subprocess.run(f"cat {temp_file} | pbcopy", shell=True, check=True)
        print(f"✅ Effects reference copied to clipboard!")
        print(f"✅ Also saved to: {output_file}")
        
        # Print summary
        lines = effects_text.split('\n')
        for line in lines[-7:-1]:  # Print summary section
            if line.strip():
                print(f"   {line}")
                
    except subprocess.CalledProcessError:
        print("❌ Failed to copy to clipboard. Make sure you're on macOS.")
        print(f"✅ But file saved to: {output_file}")
    
    # Clean up temp file
    Path(temp_file).unlink()


if __name__ == "__main__":
    main()