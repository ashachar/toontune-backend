#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert SVG to PNG using cairosvg."""

import sys
import cairosvg

def main():
    if len(sys.argv) != 3:
        print("Usage: python svg_to_png.py input.svg output.png")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    cairosvg.svg2png(url=input_file, write_to=output_file, output_width=800, output_height=800)
    print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    main()