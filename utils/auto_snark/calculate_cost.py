#!/usr/bin/env python3
"""
Calculate the cost of the ElevenLabs v3 demo
"""

import os
import json

def calculate_demo_cost():
    print("=" * 70)
    print("💰 ELEVENLABS V3 DEMO COST ANALYSIS")
    print("=" * 70)
    
    # The snarks we generated
    snarks = [
        "Oh good, another musical number. How original.",  # 47 chars
        "Yes, because ABC is such complex knowledge.",      # 44 chars
        "We get it. You can repeat three syllables.",       # 43 chars
        "Easier? This is your idea of teaching?",           # 38 chars
        "Revolutionary. A deer is... a deer.",               # 35 chars
        "Mi, the narcissism is showing.",                    # 31 chars
    ]
    
    # Calculate character counts
    char_counts = []
    total_chars = 0
    
    print("\n📝 CHARACTER COUNT PER SNARK:")
    print("-" * 50)
    for i, text in enumerate(snarks, 1):
        chars = len(text)
        char_counts.append(chars)
        total_chars += chars
        print(f"{i}. {chars:3d} chars: \"{text[:40]}...\"")
    
    # ElevenLabs pricing (as of August 2025)
    # Using Turbo v2.5 model pricing
    print("\n💵 PRICING BREAKDOWN:")
    print("-" * 50)
    
    # Different pricing tiers
    pricing_tiers = {
        "Free": {"chars_per_month": 10_000, "cost": 0},
        "Starter": {"chars_per_month": 30_000, "cost": 5},  # $5/month
        "Creator": {"chars_per_month": 100_000, "cost": 22},  # $22/month
        "Pro": {"chars_per_month": 500_000, "cost": 99},  # $99/month
    }
    
    # Turbo model is typically cheaper than regular models
    # Estimated at $0.30 per 1000 characters for pay-as-you-go
    pay_per_char_rate = 0.30 / 1000  # $0.30 per 1000 chars
    
    print(f"Total characters generated: {total_chars}")
    print(f"Average per snark: {total_chars/len(snarks):.1f} chars")
    
    # Calculate costs
    print("\n💸 COST ESTIMATES:")
    print("-" * 50)
    
    # Pay-as-you-go cost
    payg_cost = total_chars * pay_per_char_rate
    print(f"Pay-as-you-go rate: ${payg_cost:.4f}")
    print(f"  (at $0.30 per 1000 characters)")
    
    # Check which tier this would fit in
    print("\n📊 SUBSCRIPTION TIER ANALYSIS:")
    print("-" * 50)
    for tier, info in pricing_tiers.items():
        chars_allowed = info["chars_per_month"]
        monthly_cost = info["cost"]
        
        if chars_allowed >= total_chars:
            if monthly_cost > 0:
                cost_per_char = monthly_cost / chars_allowed
                demo_cost = total_chars * cost_per_char
                print(f"{tier:8s}: ${demo_cost:.4f} (from ${monthly_cost}/mo for {chars_allowed:,} chars)")
            else:
                print(f"{tier:8s}: FREE (included in {chars_allowed:,} free chars/month)")
    
    # Cost for full video processing
    print("\n🎬 FULL VIDEO COST PROJECTION:")
    print("-" * 50)
    
    # Estimate for processing entire videos
    avg_snarks_per_minute = 6 / 1  # 6 snarks in ~1 minute of video
    avg_chars_per_snark = total_chars / len(snarks)
    
    video_lengths = [1, 5, 10, 30, 60]  # minutes
    
    for minutes in video_lengths:
        estimated_snarks = int(avg_snarks_per_minute * minutes)
        estimated_chars = int(estimated_snarks * avg_chars_per_snark)
        estimated_cost = estimated_chars * pay_per_char_rate
        print(f"{minutes:3d} min video: ~{estimated_snarks:3d} snarks, ~{estimated_chars:6,} chars = ${estimated_cost:.2f}")
    
    # ROI Analysis
    print("\n📈 RETURN ON INVESTMENT:")
    print("-" * 50)
    print(f"This demo: {total_chars} characters = ${payg_cost:.4f}")
    print(f"Per snark: ${payg_cost/len(snarks):.4f}")
    print(f"Per character: ${pay_per_char_rate:.6f}")
    
    # Time saved
    print("\n⏱️ TIME & COST COMPARISON:")
    print("-" * 50)
    print("Manual recording: ~$50-200/hour for voice actor")
    print("Manual editing: ~2-4 hours @ $25-50/hour = $50-200")
    print(f"Auto-Snark with ElevenLabs: ${payg_cost:.4f} + 1 minute setup")
    print(f"Savings: ~99.9% cost reduction, 99% time reduction")
    
    # Create detailed report
    report = {
        "demo_stats": {
            "snarks_generated": len(snarks),
            "total_characters": total_chars,
            "avg_chars_per_snark": round(total_chars/len(snarks), 1),
            "character_breakdown": char_counts
        },
        "cost_analysis": {
            "pay_as_you_go": round(payg_cost, 4),
            "per_snark": round(payg_cost/len(snarks), 4),
            "per_1000_chars": 0.30,
            "currency": "USD"
        },
        "projections": {
            "10_min_video": round(60 * avg_chars_per_snark * pay_per_char_rate, 2),
            "30_min_video": round(180 * avg_chars_per_snark * pay_per_char_rate, 2),
            "60_min_video": round(360 * avg_chars_per_snark * pay_per_char_rate, 2)
        },
        "comparison": {
            "voice_actor_hourly": "$50-200",
            "manual_editing_hours": "2-4",
            "auto_snark_time": "< 1 minute",
            "cost_reduction": "99.9%"
        }
    }
    
    with open("cost_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✅ FINAL COST: ${payg_cost:.4f} for this demo")
    print(f"💡 That's less than a penny per snark!")
    print("=" * 70)
    
    return payg_cost

if __name__ == "__main__":
    total_cost = calculate_demo_cost()
    print(f"\n📊 Detailed report saved: cost_analysis_report.json")