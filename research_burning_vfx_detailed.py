#!/usr/bin/env python3
"""
Detailed research of burning VFX on productioncrate.com
Focus on capturing actual effect previews and technical details
"""

import asyncio
import os
from playwright.async_api import async_playwright

async def detailed_vfx_research():
    """Research burning VFX with focus on preview captures and technical specs"""
    
    # Create output directory
    output_dir = "burning_vfx_detailed"
    os.makedirs(output_dir, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=2000,
            args=['--start-maximized']
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        
        page = await context.new_page()
        
        try:
            print("Navigating to productioncrate.com...")
            
            # Start with general fire/burning search
            await page.goto("https://www.productioncrate.com", wait_until='networkidle')
            await page.wait_for_timeout(3000)
            
            # Look for search box and search for fire effects
            search_box = await page.query_selector('input[type="search"], input[placeholder*="search"], .search-input')
            if search_box:
                await search_box.fill("fire burning letters text")
                await search_box.press('Enter')
                await page.wait_for_timeout(5000)
            
            # Capture search results
            await page.screenshot(
                path=f"{output_dir}/fire_search_results.png",
                full_page=True
            )
            
            # Look for VFX category filter
            vfx_filter = await page.query_selector('text="VFX"')
            if vfx_filter:
                await vfx_filter.click()
                await page.wait_for_timeout(3000)
                
                await page.screenshot(
                    path=f"{output_dir}/vfx_filtered_results.png",
                    full_page=True
                )
            
            # Find video/effect containers with previews
            print("Looking for effect containers...")
            
            # Multiple selector strategies for effect cards
            selectors = [
                '.asset-card',
                '.video-card', 
                '.effect-card',
                '[data-video]',
                '.preview-container',
                '.thumbnail',
                'div[class*="card"]',
                'div[class*="item"]'
            ]
            
            effect_elements = []
            for selector in selectors:
                elements = await page.query_selector_all(selector)
                if elements:
                    effect_elements.extend(elements)
                    print(f"Found {len(elements)} elements with selector: {selector}")
                    break
            
            if not effect_elements:
                # Fallback: look for any clickable elements with images
                effect_elements = await page.query_selector_all('a img, div img, video')
            
            print(f"Found {len(effect_elements)} potential effects to examine")
            
            # Examine top effects in detail
            detailed_effects = []
            
            for i, element in enumerate(effect_elements[:8]):  # Check top 8
                try:
                    print(f"\nExamining effect {i+1}...")
                    
                    # Scroll element into view
                    await element.scroll_into_view_if_needed()
                    await page.wait_for_timeout(2000)
                    
                    # Get parent container for more context
                    parent = await element.query_selector('xpath=..')
                    if parent:
                        container = parent
                    else:
                        container = element
                    
                    # Look for hover effects or click to reveal preview
                    try:
                        await container.hover()
                        await page.wait_for_timeout(1500)
                    except:
                        pass
                    
                    # Capture effect preview
                    screenshot_path = f"{output_dir}/detailed_effect_{i+1:02d}.png"
                    await container.screenshot(path=screenshot_path)
                    
                    # Try to extract title and description
                    title_selectors = ['h1', 'h2', 'h3', '.title', '.name', '[class*="title"]', '[class*="name"]']
                    title = "Unknown Effect"
                    
                    for sel in title_selectors:
                        title_elem = await container.query_selector(sel)
                        if title_elem:
                            title_text = await title_elem.inner_text()
                            if title_text and len(title_text.strip()) > 0:
                                title = title_text.strip()
                                break
                    
                    # Get all text content for analysis
                    text_content = await container.inner_text()
                    
                    # Check for quality indicators
                    quality_indicators = {
                        '4K': '4k' in text_content.lower() or '4K' in text_content,
                        'HD': 'hd' in text_content.lower() or '1080' in text_content,
                        'Alpha': 'alpha' in text_content.lower() or 'transparent' in text_content.lower(),
                        'MOV': '.mov' in text_content.lower(),
                        'MP4': '.mp4' in text_content.lower(),
                        'Fire': any(word in text_content.lower() for word in ['fire', 'flame', 'burn', 'ignite']),
                        'Smoke': any(word in text_content.lower() for word in ['smoke', 'ash', 'char', 'ember']),
                        'Free': 'free' in text_content.lower(),
                        'Premium': 'premium' in text_content.lower() or 'pro' in text_content.lower()
                    }
                    
                    effect_info = {
                        'id': i+1,
                        'title': title,
                        'screenshot': screenshot_path,
                        'quality_indicators': quality_indicators,
                        'text_preview': text_content[:300]
                    }
                    
                    detailed_effects.append(effect_info)
                    print(f"Captured: {title[:50]}...")
                    
                    # Try to click for detail view
                    try:
                        link = await container.query_selector('a')
                        if link:
                            print(f"Attempting to view details...")
                            await link.click()
                            await page.wait_for_timeout(4000)
                            
                            # Capture detail page
                            detail_path = f"{output_dir}/detail_view_{i+1:02d}.png"
                            await page.screenshot(path=detail_path, full_page=True)
                            
                            # Look for technical specs
                            specs_text = await page.inner_text('body')
                            
                            # Go back
                            await page.go_back()
                            await page.wait_for_timeout(2000)
                            
                            effect_info['detail_screenshot'] = detail_path
                            effect_info['detail_specs'] = specs_text[:500]
                    except Exception as e:
                        print(f"Could not access detail view: {e}")
                
                except Exception as e:
                    print(f"Error examining effect {i+1}: {e}")
                    continue
            
            # Generate comprehensive report
            report_path = f"{output_dir}/detailed_research_report.md"
            with open(report_path, 'w') as f:
                f.write("# Burning Letter VFX Research Report\n\n")
                f.write(f"**Research Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Source:** ProductionCrate.com\n")
                f.write(f"**Search Terms:** fire burning letters text\n\n")
                
                f.write("## Summary\n\n")
                f.write(f"- Total effects examined: {len(detailed_effects)}\n")
                f.write(f"- Effects with fire/burning keywords: {sum(1 for e in detailed_effects if e['quality_indicators']['Fire'])}\n")
                f.write(f"- Effects with smoke keywords: {sum(1 for e in detailed_effects if e['quality_indicators']['Smoke'])}\n")
                f.write(f"- HD/4K effects: {sum(1 for e in detailed_effects if e['quality_indicators']['HD'] or e['quality_indicators']['4K'])}\n")
                f.write(f"- Alpha channel effects: {sum(1 for e in detailed_effects if e['quality_indicators']['Alpha'])}\n\n")
                
                f.write("## Detailed Effects Analysis\n\n")
                
                for effect in detailed_effects:
                    f.write(f"### {effect['id']}. {effect['title']}\n\n")
                    f.write(f"**Screenshot:** `{effect['screenshot']}`\n\n")
                    
                    f.write("**Quality Indicators:**\n")
                    for indicator, present in effect['quality_indicators'].items():
                        status = "✅" if present else "❌"
                        f.write(f"- {indicator}: {status}\n")
                    f.write("\n")
                    
                    f.write("**Content Preview:**\n")
                    f.write(f"```\n{effect['text_preview']}\n```\n\n")
                    
                    if 'detail_screenshot' in effect:
                        f.write(f"**Detail View:** `{effect['detail_screenshot']}`\n\n")
                    
                    f.write("---\n\n")
                
                f.write("## Recommendations for Burning Letter Animation\n\n")
                f.write("### Top Priorities:\n")
                f.write("1. **High Resolution**: Look for 4K or HD effects for crisp detail\n")
                f.write("2. **Alpha Channel**: Essential for compositing over text\n")
                f.write("3. **Multiple Stages**: Effects showing ignition → burning → charring\n")
                f.write("4. **Volumetric Elements**: Realistic smoke and ember particles\n")
                f.write("5. **Professional Quality**: Avoid amateur or low-quality effects\n\n")
                
                f.write("### Implementation Strategy:\n")
                f.write("- Download most promising effects as reference material\n")
                f.write("- Study flame behavior and color gradients\n")
                f.write("- Analyze smoke patterns and particle movement\n")
                f.write("- Note timing of different burning stages\n")
                f.write("- Consider creating custom effects based on these references\n")
            
            print(f"\nDetailed research complete!")
            print(f"Report saved: {report_path}")
            print(f"Screenshots in: {output_dir}/")
            
            # Keep browser open for manual review
            print("\nBrowser open for manual inspection. Close when done.")
            await page.wait_for_timeout(300000)  # 5 minutes
            
        except Exception as e:
            print(f"Research error: {e}")
            await page.screenshot(path=f"{output_dir}/error_capture.png", full_page=True)
            
        finally:
            await browser.close()

if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(detailed_vfx_research())