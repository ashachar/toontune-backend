#!/usr/bin/env python3
"""
Research burning letter animations on productioncrate.com
Captures screenshots of promising VFX for burning text effects
"""

import asyncio
import os
from playwright.async_api import async_playwright

async def research_burning_letters():
    """Research and capture burning letter VFX from productioncrate.com"""
    
    # Create output directory for screenshots
    output_dir = "burning_letter_research"
    os.makedirs(output_dir, exist_ok=True)
    
    async with async_playwright() as p:
        # Launch browser in headed mode for visual debugging
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=1000  # Slow down for better visibility
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        
        page = await context.new_page()
        
        try:
            print("Navigating to productioncrate.com burning letter search...")
            
            # Navigate to the burning letter search URL
            url = "https://www.productioncrate.com/search/burning%20letter?main_category=vfx"
            await page.goto(url, wait_until='networkidle')
            
            # Wait for page to fully load
            await page.wait_for_timeout(3000)
            
            # Take initial screenshot of search results
            await page.screenshot(
                path=f"{output_dir}/01_search_results_overview.png",
                full_page=True
            )
            print("Captured overview of search results")
            
            # Look for video/effect containers
            effect_containers = await page.query_selector_all('[class*="card"], [class*="item"], [class*="video"], [class*="effect"]')
            
            if not effect_containers:
                # Try alternative selectors
                effect_containers = await page.query_selector_all('div[class*="grid"] > div, .search-result, .asset-card')
            
            print(f"Found {len(effect_containers)} potential effect containers")
            
            # Capture individual effects
            promising_effects = []
            
            for i, container in enumerate(effect_containers[:10]):  # Limit to first 10
                try:
                    # Scroll to element
                    await container.scroll_into_view_if_needed()
                    await page.wait_for_timeout(1000)
                    
                    # Take screenshot of individual effect
                    screenshot_path = f"{output_dir}/effect_{i+1:02d}.png"
                    await container.screenshot(path=screenshot_path)
                    
                    # Try to get title/description
                    title_elem = await container.query_selector('h3, .title, [class*="title"], [class*="name"]')
                    title = await title_elem.inner_text() if title_elem else f"Effect {i+1}"
                    
                    # Check for quality indicators
                    quality_text = await container.inner_text()
                    has_4k = "4K" in quality_text or "4k" in quality_text
                    has_hd = "HD" in quality_text or "1080" in quality_text
                    has_alpha = "alpha" in quality_text.lower() or "transparent" in quality_text.lower()
                    
                    effect_info = {
                        'index': i+1,
                        'title': title.strip(),
                        'screenshot': screenshot_path,
                        'has_4k': has_4k,
                        'has_hd': has_hd,
                        'has_alpha': has_alpha,
                        'text_content': quality_text[:200]  # First 200 chars
                    }
                    
                    promising_effects.append(effect_info)
                    print(f"Captured effect {i+1}: {title[:50]}...")
                    
                except Exception as e:
                    print(f"Error capturing effect {i+1}: {e}")
                    continue
            
            # Look for specific burning/fire related effects
            print("\nSearching for fire-specific elements...")
            fire_keywords = ['fire', 'flame', 'burn', 'smoke', 'char', 'ash', 'ember', 'ignite']
            
            for keyword in fire_keywords:
                elements = await page.query_selector_all(f'text="{keyword}"')
                if elements:
                    print(f"Found {len(elements)} elements mentioning '{keyword}'")
            
            # Try to find and click on a promising effect for detail view
            if promising_effects:
                print(f"\nAttempting to view details of first effect...")
                first_container = effect_containers[0]
                
                # Look for clickable link or button
                link = await first_container.query_selector('a, button, [role="button"]')
                if link:
                    try:
                        await link.click()
                        await page.wait_for_timeout(3000)
                        
                        # Capture detail view
                        await page.screenshot(
                            path=f"{output_dir}/detail_view_example.png",
                            full_page=True
                        )
                        print("Captured detail view of first effect")
                        
                        # Go back to search results
                        await page.go_back()
                        await page.wait_for_timeout(2000)
                        
                    except Exception as e:
                        print(f"Could not open detail view: {e}")
            
            # Generate research report
            report_path = f"{output_dir}/research_report.txt"
            with open(report_path, 'w') as f:
                f.write("BURNING LETTER VFX RESEARCH REPORT\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Search URL: {url}\n")
                f.write(f"Total effects captured: {len(promising_effects)}\n\n")
                
                f.write("CAPTURED EFFECTS:\n")
                f.write("-" * 20 + "\n")
                
                for effect in promising_effects:
                    f.write(f"\n{effect['index']}. {effect['title']}\n")
                    f.write(f"   Screenshot: {effect['screenshot']}\n")
                    f.write(f"   4K Quality: {'Yes' if effect['has_4k'] else 'No'}\n")
                    f.write(f"   HD Quality: {'Yes' if effect['has_hd'] else 'No'}\n")
                    f.write(f"   Alpha Channel: {'Yes' if effect['has_alpha'] else 'Unknown'}\n")
                    f.write(f"   Preview: {effect['text_content'][:100]}...\n")
                
                f.write("\nRECOMMENDATIONS:\n")
                f.write("-" * 15 + "\n")
                f.write("Look for effects with:\n")
                f.write("- High resolution (4K/HD)\n")
                f.write("- Alpha channel support\n")
                f.write("- Multiple flame stages\n")
                f.write("- Volumetric smoke\n")
                f.write("- Realistic char/ash effects\n")
                f.write("- Professional VFX quality\n")
            
            print(f"\nResearch complete! Generated report: {report_path}")
            print(f"Screenshots saved in: {output_dir}/")
            
            # Keep browser open for manual inspection
            print("\nBrowser staying open for manual inspection...")
            print("Close the browser window when you're done reviewing the results.")
            await page.wait_for_timeout(300000)  # Wait 5 minutes for manual review
            
        except Exception as e:
            print(f"Error during research: {e}")
            await page.screenshot(path=f"{output_dir}/error_screenshot.png")
            
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(research_burning_letters())