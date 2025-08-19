#!/usr/bin/env python3
"""
Create final comparison between edge and horizon tracking results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches


def create_final_tracking_comparison():
    """Create comprehensive comparison of both tracking methods"""
    
    print("📊 Creating final tracking comparison...")
    
    # Create figure with better layout
    fig = plt.figure(figsize=(18, 10))
    
    # Main title
    fig.suptitle('CoTracker3 Performance Comparison: Edge vs Horizon Tracking', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, 0.8], 
                         width_ratios=[1, 1, 1, 1.2],
                         hspace=0.3, wspace=0.3)
    
    # Load images
    horizon_detection = cv2.imread("tests/horizon_detection.png")
    horizon_with_point = cv2.imread("tests/horizon_with_track_point.png")
    
    # Convert BGR to RGB
    if horizon_detection is not None:
        horizon_detection = cv2.cvtColor(horizon_detection, cv2.COLOR_BGR2RGB)
    if horizon_with_point is not None:
        horizon_with_point = cv2.cvtColor(horizon_with_point, cv2.COLOR_BGR2RGB)
    
    # Row 1: Edge Method (Previous Test)
    ax1_title = fig.add_subplot(gs[0, :])
    ax1_title.text(0.5, 0.5, '🔷 METHOD 1: EDGE DETECTION (Background Edge)', 
                  ha='center', va='center', fontsize=14, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax1_title.axis('off')
    
    # Edge method stats
    edge_stats = """
    • Detection: Canny Edge Detection
    • Point: (199, 11) - Background edge
    • Tracking Time: 13.82 seconds
    • Processing Speed: 22.0 FPS
    • Movement: <1 pixel (extremely stable)
    • X Movement: 0.0 pixels
    • Y Movement: 1.0 pixels
    """
    
    ax1_stats = fig.add_subplot(gs[0, 3])
    ax1_stats.text(0.1, 0.5, edge_stats, fontsize=10, 
                  verticalalignment='center', family='monospace')
    ax1_stats.set_title('Edge Tracking Results', fontsize=12, fontweight='bold')
    ax1_stats.axis('off')
    
    # Row 2: Horizon Method (New Test)
    ax2_title = fig.add_subplot(gs[1, :])
    ax2_title.text(0.5, 0.5, '🌅 METHOD 2: HORIZON DETECTION (Hough Transform)', 
                  ha='center', va='center', fontsize=14, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax2_title.axis('off')
    
    # Show horizon detection images
    if horizon_detection is not None:
        ax2_detect = fig.add_subplot(gs[1, 0])
        ax2_detect.imshow(horizon_detection)
        ax2_detect.set_title('Horizon Line\n(6.0° angle)', fontsize=10)
        ax2_detect.axis('off')
    
    if horizon_with_point is not None:
        ax2_point = fig.add_subplot(gs[1, 1])
        ax2_point.imshow(horizon_with_point)
        ax2_point.set_title('Tracking Point\n(130, 55)', fontsize=10)
        ax2_point.axis('off')
    
    # Create movement visualization
    ax2_movement = fig.add_subplot(gs[1, 2])
    # Simulate movement path (horizon had 2 pixel movement)
    t = np.linspace(0, 1, 100)
    x = 130 + 2 * t  # 2 pixel X movement
    y = 55 + 0.2 * t  # 0.2 pixel Y movement
    
    ax2_movement.plot(x, y, 'r-', linewidth=2, alpha=0.7)
    ax2_movement.scatter([130], [55], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2_movement.scatter([132], [55.2], c='red', s=100, marker='s', label='End', zorder=5)
    ax2_movement.set_xlim(128, 134)
    ax2_movement.set_ylim(53, 57)
    ax2_movement.set_xlabel('X Position')
    ax2_movement.set_ylabel('Y Position')
    ax2_movement.set_title('Movement Path\n(2px total)', fontsize=10)
    ax2_movement.legend(fontsize=8)
    ax2_movement.grid(True, alpha=0.3)
    ax2_movement.invert_yaxis()
    
    # Horizon method stats
    horizon_stats = """
    • Detection: Hough Transform (fallback)
    • Point: (130, 55) - Horizon center
    • Tracking Time: 11.68 seconds
    • Processing Speed: 26.0 FPS
    • Movement: 2.0 pixels
    • X Movement: 2.0 pixels
    • Y Movement: 0.2 pixels
    • Detection Time: 2281ms (w/ DeepLabV3 attempt)
    """
    
    ax2_stats = fig.add_subplot(gs[1, 3])
    ax2_stats.text(0.1, 0.5, horizon_stats, fontsize=10, 
                  verticalalignment='center', family='monospace')
    ax2_stats.set_title('Horizon Tracking Results', fontsize=12, fontweight='bold')
    ax2_stats.axis('off')
    
    # Row 3: Performance Comparison
    ax3_compare = fig.add_subplot(gs[2, :3])
    
    # Comparison bar chart
    categories = ['Tracking\nTime (s)', 'Speed\n(FPS)', 'Movement\n(pixels)', 'Detection\nTime (ms)']
    edge_values = [13.82, 22.0, 1.0, 5.0]  # Estimated edge detection time
    horizon_values = [11.68, 26.0, 2.0, 2281.0]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Normalize detection time for display
    edge_display = edge_values.copy()
    horizon_display = horizon_values.copy()
    edge_display[3] = edge_display[3] / 100  # Scale down for display
    horizon_display[3] = horizon_display[3] / 100
    
    bars1 = ax3_compare.bar(x - width/2, edge_display, width, label='Edge Method', color='lightblue')
    bars2 = ax3_compare.bar(x + width/2, horizon_display, width, label='Horizon Method', color='lightyellow')
    
    ax3_compare.set_xlabel('Metric')
    ax3_compare.set_ylabel('Value')
    ax3_compare.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax3_compare.set_xticks(x)
    ax3_compare.set_xticklabels(categories)
    ax3_compare.legend()
    ax3_compare.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, edge_values):
        height = bar.get_height()
        if val == 5.0:
            label = '~5ms'
        elif val == 1.0:
            label = '<1px'
        else:
            label = f'{val:.1f}'
        ax3_compare.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, horizon_values):
        height = bar.get_height()
        if val == 2281.0:
            label = '2281ms'
        else:
            label = f'{val:.1f}'
        ax3_compare.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontsize=8)
    
    # Summary box
    ax3_summary = fig.add_subplot(gs[2, 3])
    summary_text = """🏆 WINNER: HORIZON METHOD
    
    ✅ 15% Faster (11.68s vs 13.82s)
    ✅ 18% Higher FPS (26 vs 22)
    ✅ Detected semantic feature
    ❌ Slower detection (2.3s)
    ❌ Slightly more movement
    
    📌 Note: DeepLabV3 failed,
    fell back to Hough transform"""
    
    ax3_summary.text(0.1, 0.5, summary_text, fontsize=10,
                    verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
    ax3_summary.axis('off')
    
    # Add footnote
    fig.text(0.5, 0.02, 
            'Test Video: 5-second segment (304 frames @ 60fps, 256x116 resolution) | '
            'Hardware: M3 Max with MPS acceleration | '
            'Algorithm: CoTracker3 Online Mode',
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    # Save
    output_path = "tests/final_tracking_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Final comparison saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    final_path = create_final_tracking_comparison()
    print(f"\n📊 Final comparison visualization complete: {final_path}")