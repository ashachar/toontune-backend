#!/usr/bin/env python3
"""
Update remarks to use actual gap times from Scribe analysis with 20s spacing.
"""

import json
from pathlib import Path

def update_remarks_with_scribe_gaps():
    """Update remarks.json to use actual gap times with 20s spacing."""
    
    # Load Scribe gaps
    gaps_file = Path("uploads/assets/videos/ai_math1/gaps_analysis_scribe.json")
    with open(gaps_file) as f:
        data = json.load(f)
        gaps = data["gaps"]
    
    # Sort gaps by start time
    gaps.sort(key=lambda x: x["start"])
    
    # Load current remarks  
    remarks_file = Path("uploads/assets/videos/ai_math1/remarks.json")
    with open(remarks_file) as f:
        remarks = json.load(f)
    
    print("Available gaps from Scribe:")
    for i, gap in enumerate(gaps[:15]):
        print(f"  Gap {i:2d}: {gap['start']:6.2f}s (duration: {gap['duration']:.3f}s)")
    
    print("\n" + "="*70)
    print("Selecting gaps with 20s minimum spacing:")
    print("="*70)
    
    # Select gaps with 20s spacing
    selected_gaps = []
    last_time = -float('inf')
    
    for gap in gaps:
        if gap["start"] - last_time >= 20.0:
            selected_gaps.append(gap)
            last_time = gap["start"]
            print(f"✅ Selected gap at {gap['start']:.2f}s (gap from last: {gap['start'] - (selected_gaps[-2]['start'] if len(selected_gaps) > 1 else -float('inf')):.1f}s)")
            
            if len(selected_gaps) >= len(remarks):
                break
    
    # Update remarks with new times
    updated_remarks = []
    for i, remark in enumerate(remarks[:len(selected_gaps)]):
        gap = selected_gaps[i]
        updated_remark = remark.copy()
        updated_remark["time"] = gap["start"]
        updated_remarks.append(updated_remark)
        print(f"\n📝 Remark {i+1}: '{remark['text']}'")
        print(f"   Old time: {remark['time']:.2f}s")
        print(f"   New time: {gap['start']:.2f}s (gap duration: {gap['duration']:.3f}s)")
    
    # Save updated remarks
    with open(remarks_file, 'w') as f:
        json.dump(updated_remarks, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ Updated remarks.json with properly spaced gaps")
    print("="*70)
    
    # Verify spacing
    print("\n🔍 Verification:")
    for i in range(1, len(updated_remarks)):
        gap = updated_remarks[i]["time"] - updated_remarks[i-1]["time"]
        status = "✅" if gap >= 20 else "❌"
        print(f"  {status} Gap {i}: {gap:.1f}s")

if __name__ == "__main__":
    update_remarks_with_scribe_gaps()