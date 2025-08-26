#!/usr/bin/env python3
"""
Regenerate remarks.json with 20-second minimum spacing between comments.
"""

import json
from pathlib import Path
import sys

def filter_remarks_with_spacing(remarks_file: str, min_gap: float = 20.0) -> str:
    """Filter remarks to ensure minimum spacing between consecutive comments."""
    
    # Load existing remarks
    with open(remarks_file) as f:
        remarks = json.load(f)
    
    if not remarks:
        return remarks_file
    
    # Sort by time
    remarks.sort(key=lambda r: r["time"])
    
    # Filter with minimum gap constraint
    filtered = []
    last_time = -float('inf')
    
    print(f"Original remarks: {len(remarks)}")
    print(f"Enforcing minimum gap: {min_gap}s\n")
    
    for remark in remarks:
        time_gap = remark["time"] - last_time
        
        if time_gap >= min_gap:
            filtered.append(remark)
            last_time = remark["time"]
            print(f"✅ Keeping remark at {remark['time']:.1f}s: '{remark['text']}' (gap: {time_gap:.1f}s)")
        else:
            print(f"❌ Skipping remark at {remark['time']:.1f}s: '{remark['text']}' (gap: {time_gap:.1f}s < {min_gap}s)")
    
    # Save filtered remarks
    output_file = remarks_file.replace('.json', '_spaced.json')
    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=2)
    
    print(f"\n✅ Filtered remarks: {len(filtered)} (removed {len(remarks) - len(filtered)})")
    print(f"📄 Saved to: {output_file}")
    
    # Verify spacing
    print("\n🔍 Verification:")
    for i in range(1, len(filtered)):
        gap = filtered[i]["time"] - filtered[i-1]["time"]
        print(f"  Gap {i}: {gap:.1f}s {'✅' if gap >= min_gap else '❌ ERROR'}")
    
    return output_file

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        remarks_file = sys.argv[1]
    else:
        # Default to ai_math1 remarks
        remarks_file = "uploads/assets/videos/ai_math1/remarks.json"
    
    if not Path(remarks_file).exists():
        print(f"❌ File not found: {remarks_file}")
        sys.exit(1)
    
    # Filter with 20-second spacing
    output_file = filter_remarks_with_spacing(remarks_file, min_gap=20.0)
    
    # Also create a backup of original
    backup_file = remarks_file.replace('.json', '_original.json')
    if not Path(backup_file).exists():
        import shutil
        shutil.copy(remarks_file, backup_file)
        print(f"\n📋 Original backed up to: {backup_file}")
    
    # Optionally replace the original file
    print(f"\n💡 To use the spaced version, run:")
    print(f"   cp {output_file} {remarks_file}")

if __name__ == "__main__":
    main()