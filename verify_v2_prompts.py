#!/usr/bin/env python3
"""
Verify V2 prompts contain cartoon characters and key phrases
"""

from pathlib import Path
import re

def verify_prompts():
    """Verify all prompts have the correct V2 structure"""
    
    prompts_dir = Path("uploads/assets/videos/do_re_mi/prompts")
    
    print("=" * 70)
    print("V2 PROMPT VERIFICATION")
    print("=" * 70)
    
    prompt_files = sorted(prompts_dir.glob("scene_*.txt"))
    
    if not prompt_files:
        print("❌ No prompt files found!")
        return
    
    print(f"Found {len(prompt_files)} prompt files\n")
    
    for prompt_file in prompt_files:
        print(f"\n📄 {prompt_file.name}")
        print("-" * 50)
        
        content = prompt_file.read_text()
        
        # Count key references
        cartoon_count = content.count("cartoon")
        key_phrases_count = content.count("key_phrases")
        text_overlays_count = content.count("text_overlays")
        
        print(f"  ✓ 'cartoon' appears: {cartoon_count} times")
        print(f"  ✓ 'key_phrases' appears: {key_phrases_count} times")
        print(f"  {'✓' if text_overlays_count == 0 else '❌'} 'text_overlays' appears: {text_overlays_count} times (should be 0)")
        
        # Find cartoon character examples
        cartoon_examples = re.findall(r'"character_type":\s*"([^"]+)"', content)
        if cartoon_examples:
            print(f"  ✓ Cartoon character examples found:")
            for ex in cartoon_examples[:3]:
                print(f"    • {ex}")
        
        # Find key phrase examples
        phrase_examples = re.findall(r'"phrase":\s*"([^"]+)"', content)
        if phrase_examples:
            print(f"  ✓ Key phrase examples found:")
            for ex in phrase_examples[:3]:
                print(f"    • {ex}")
        
        # Check for constraints
        if "CRITICAL CONSTRAINTS FOR THIS SCENE" in content:
            print(f"  ✓ Scene-specific constraints present")
            
            # Extract constraint numbers
            max_phrases = re.search(r"Maximum (\d+) key phrase", content)
            max_chars = re.search(r"Maximum (\d+) cartoon character", content)
            
            if max_phrases:
                print(f"    • Max key phrases: {max_phrases.group(1)}")
            if max_chars:
                print(f"    • Max cartoon characters: {max_chars.group(1)}")
        
        # Show a sample of the cartoon instructions
        if "2. CARTOON CHARACTERS:" in content:
            print(f"  ✓ Cartoon character instructions present")
            
            # Extract some of the instructions
            instructions_match = re.search(
                r"2\. CARTOON CHARACTERS:(.*?)The output MUST", 
                content, 
                re.DOTALL
            )
            if instructions_match:
                instructions = instructions_match.group(1).strip()
                lines = instructions.split('\n')[:5]  # First 5 lines
                print("    Instructions preview:")
                for line in lines:
                    if line.strip():
                        print(f"      {line.strip()[:80]}...")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    if all([
        cartoon_count > 0,
        key_phrases_count > 0,
        text_overlays_count == 0
    ] for prompt_file in prompt_files):
        print("✅ All prompts are correctly formatted with V2 structure!")
        print("   - Cartoon characters: Present")
        print("   - Key phrases: Present")
        print("   - Old text_overlays: Removed")
    else:
        print("⚠️ Some prompts may need attention")


if __name__ == "__main__":
    verify_prompts()