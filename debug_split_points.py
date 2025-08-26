#!/usr/bin/env python3
"""
Debug split points to understand segment overlap issue.
"""

import json
from pathlib import Path

def main():
    # Load remarks
    remarks_path = Path("uploads/assets/videos/ai_math1/remarks.json")
    with open(remarks_path) as f:
        remarks = json.load(f)
    
    # Load gaps from Scribe
    gaps_path = Path("uploads/assets/videos/ai_math1/gaps_analysis_scribe.json")
    with open(gaps_path) as f:
        data = json.load(f)
        gaps = data["gaps"]
    
    print("="*70)
    print("SPLIT POINT DEBUG ANALYSIS")
    print("="*70)
    
    print(f"\nTotal remarks: {len(remarks)}")
    print(f"Total gaps detected: {len(gaps)}")
    
    print("\nFirst 10 gaps (chronological):")
    for i, gap in enumerate(gaps[:10]):
        print(f"  Gap {i}: {gap['start']:6.2f}s - {gap['end']:6.2f}s (duration: {gap['duration']:.3f}s)")
    
    print("\nRemarks to be inserted:")
    for i, remark in enumerate(remarks):
        print(f"  Remark {i+1}: '{remark['text']}' at {remark.get('time', 'unknown')}s")
    
    # Simulate the assignment logic
    print("\n" + "="*70)
    print("SIMULATED GAP ASSIGNMENT (chronological)")
    print("="*70)
    
    gaps_sorted = sorted(gaps, key=lambda x: x["start"])[:10]  # First 10 gaps chronologically
    
    for i, (remark, gap) in enumerate(zip(remarks, gaps_sorted)):
        print(f"\nRemark {i+1}: '{remark['text']}'")
        print(f"  Assigned to gap: {gap['start']:.2f}s - {gap['end']:.2f}s")
        print(f"  Context before: ...{gap['context_before'][-30:]}")
        print(f"  Context after: {gap['context_after'][:30]}...")

if __name__ == "__main__":
    main()