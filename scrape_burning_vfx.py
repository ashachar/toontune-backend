#!/usr/bin/env python3
"""
Playwright script to scrape burning letter VFX from productioncrate.com
Runs in headless mode and captures screenshots of available assets.
"""

import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright

async def scrape_burning_vfx():
    """Scrape burning letter VFX assets from productioncrate.com"""
    
    # Create output directory
    output_dir = Path("vfx_screenshots")
    output_dir.mkdir(exist_ok=True)
    
    async with async_playwright() as p:
        # Launch browser in headless mode
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = await context.new_page()
        
        try:
            print("Navigating to productioncrate.com burning letters search...")
            
            # Navigate to the burning letters search page
            url = "https://www.productioncrate.com/search/burning%20letter?main_category=vfx"
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for initial page load
            await page.wait_for_timeout(3000)
            
            # Handle cookie consent if present
            try:
                cookie_button = page.locator('button:has-text("Accept"), button:has-text("OK"), button:has-text("I understand")')
                if await cookie_button.count() > 0:
                    print("Dismissing cookie consent...")
                    await cookie_button.first.click()
                    await page.wait_for_timeout(1000)
            except:
                pass
            
            # Handle any popups or modals
            try:
                close_buttons = page.locator('button[aria-label="Close"], .modal-close, .close-btn, [data-dismiss="modal"]')
                if await close_buttons.count() > 0:
                    print("Closing popups...")
                    await close_buttons.first.click()
                    await page.wait_for_timeout(1000)
            except:
                pass
            
            # Capture full page screenshot
            print("Capturing full page screenshot...")
            await page.screenshot(
                path=output_dir / "burning_letters_full_page.png",
                full_page=True
            )
            
            # Wait for content to load
            await page.wait_for_timeout(5000)
            
            # Try to find VFX cards/items
            vfx_selectors = [
                '.search-result-item',
                '.vfx-item',
                '.asset-card',
                '.product-card',
                '.grid-item',
                '[class*="card"]',
                '[class*="item"]',
                '[class*="product"]'
            ]
            
            vfx_items = None
            for selector in vfx_selectors:
                items = page.locator(selector)
                count = await items.count()
                if count > 0:
                    print(f"Found {count} items using selector: {selector}")
                    vfx_items = items
                    break
            
            if vfx_items:
                # Capture individual VFX cards
                count = await vfx_items.count()
                print(f"Capturing {count} individual VFX items...")
                
                for i in range(min(count, 20)):  # Limit to first 20 items
                    try:
                        item = vfx_items.nth(i)
                        await item.scroll_into_view_if_needed()
                        await page.wait_for_timeout(500)
                        
                        # Get item title/name for filename
                        item_name = f"vfx_item_{i+1:02d}"
                        try:
                            title_elem = item.locator('h3, .title, .name, [class*="title"], [class*="name"]').first
                            if await title_elem.count() > 0:
                                title_text = await title_elem.text_content()
                                if title_text:
                                    # Clean filename
                                    item_name = "".join(c for c in title_text[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
                                    item_name = item_name.replace(' ', '_').lower()
                        except:
                            pass
                        
                        # Screenshot individual item
                        await item.screenshot(path=output_dir / f"{item_name}_{i+1:02d}.png")
                        print(f"  Captured: {item_name}_{i+1:02d}.png")
                        
                    except Exception as e:
                        print(f"  Failed to capture item {i+1}: {e}")
            
            # Look for video previews or thumbnails
            print("Looking for video previews...")
            video_selectors = [
                'video',
                '.video-preview',
                '.thumbnail',
                '[class*="video"]',
                '[class*="preview"]',
                'img[src*="video"], img[src*="preview"]'
            ]
            
            for selector in video_selectors:
                elements = page.locator(selector)
                count = await elements.count()
                if count > 0:
                    print(f"Found {count} video elements using: {selector}")
                    
                    # Capture area around video elements
                    for i in range(min(count, 10)):
                        try:
                            element = elements.nth(i)
                            await element.scroll_into_view_if_needed()
                            await page.wait_for_timeout(1000)
                            await element.screenshot(path=output_dir / f"video_preview_{i+1:02d}.png")
                            print(f"  Captured video preview {i+1}")
                        except Exception as e:
                            print(f"  Failed to capture video preview {i+1}: {e}")
            
            # Capture viewport screenshot of current view
            print("Capturing current viewport...")
            await page.screenshot(path=output_dir / "burning_letters_viewport.png")
            
            # Try to scroll and capture more content
            print("Scrolling to load more content...")
            await page.evaluate("window.scrollTo(0, window.innerHeight)")
            await page.wait_for_timeout(3000)
            await page.screenshot(path=output_dir / "burning_letters_scrolled.png")
            
            # Get page info
            title = await page.title()
            url_final = page.url
            
            print(f"\nPage scraped successfully:")
            print(f"Title: {title}")
            print(f"Final URL: {url_final}")
            print(f"Screenshots saved to: {output_dir.absolute()}")
            
            # List all created files
            print(f"\nGenerated files:")
            for file in sorted(output_dir.glob("*.png")):
                print(f"  {file.name} ({file.stat().st_size // 1024} KB)")
                
        except Exception as e:
            print(f"Error during scraping: {e}")
            # Capture error screenshot
            try:
                await page.screenshot(path=output_dir / "error_screenshot.png")
            except:
                pass
        
        finally:
            await browser.close()

if __name__ == "__main__":
    print("Starting VFX scraping in headless mode...")
    asyncio.run(scrape_burning_vfx())
    print("Scraping completed!")