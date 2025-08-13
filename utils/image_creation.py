#!/usr/bin/env python3
"""
ChatGPT Image Creation Automation Script
Automates the process of submitting images and prompts to ChatGPT
"""

import pyautogui
import time
import subprocess
import random
from datetime import datetime
import pyperclip

# Configure pyautogui safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

def read_prompts(file_path):
    """Read prompts from file and return as list"""
    try:
        with open(file_path, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {file_path}")
        return prompts
    except FileNotFoundError:
        print(f"Error: Prompts file not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error reading prompts file: {e}")
        return []

def copy_image_to_clipboard(image_path):
    """Copy image to clipboard using osascript"""
    try:
        # Use osascript to copy image to clipboard
        cmd = [
            'osascript', '-e',
            'on run argv',
            '-e', 'set the clipboard to (read (POSIX file (item 1 of argv)) as JPEG picture)',
            '-e', 'end run',
            image_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully copied image to clipboard: {image_path}")
            return True
        else:
            print(f"Error copying image to clipboard: {result.stderr}")
            return False
    except Exception as e:
        print(f"Exception copying image to clipboard: {e}")
        return False

def open_chatgpt_in_chrome():
    """Open ChatGPT in Chrome using AppleScript"""
    try:
        applescript = '''
        tell application "Google Chrome"
            activate
            open location "https://chatgpt.com/"
            delay 1
            set bounds of front window to {0, 0, 1440, 900}
        end tell
        '''
        subprocess.run(['osascript', '-e', applescript])
        print("Opened ChatGPT in Chrome")
        return True
    except Exception as e:
        print(f"Error opening ChatGPT: {e}")
        return False

def clear_textbox():
    """Clear any existing text in the textbox"""
    # Select all text
    pyautogui.hotkey('command', 'a')
    time.sleep(0.5)
    # Delete selected text
    pyautogui.press('backspace')
    time.sleep(0.5)

def paste_from_clipboard():
    """Paste content from clipboard"""
    pyautogui.hotkey('command', 'v')

def start_new_chat():
    """Start a new chat using keyboard shortcut"""
    pyautogui.hotkey('command', 'shift', 'o')

def is_night_time():
    """Check if current time is between midnight and 7 AM"""
    current_hour = datetime.now().hour
    return 0 <= current_hour < 7

def get_random_wait(min_seconds, max_seconds):
    """Get random wait time between min and max seconds"""
    return random.uniform(min_seconds, max_seconds)

def should_take_hourly_break(start_time, last_break_time):
    """Check if it's time for an hourly break"""
    current_time = time.time()
    time_since_start = (current_time - start_time) / 3600  # hours
    time_since_break = (current_time - last_break_time) / 3600  # hours
    
    # Take a break every 1-3 hours randomly
    break_interval = random.uniform(1, 3)
    return time_since_break >= break_interval

def main():
    """Main automation function"""
    # File paths
    prompts_file = "/Users/amirshachar/Desktop/image_prompts.txt"
    image_file = "/Users/amirshachar/Downloads/man.png"
    
    # Read prompts
    prompts = read_prompts(prompts_file)
    if not prompts:
        print("No prompts found. Exiting.")
        return
    
    # Copy image to clipboard
    if not copy_image_to_clipboard(image_file):
        print("Failed to copy image to clipboard. Exiting.")
        return
    
    # Open ChatGPT in Chrome
    if not open_chatgpt_in_chrome():
        print("Failed to open ChatGPT. Exiting.")
        return
    
    # Wait for page to load
    print("Waiting for ChatGPT to load...")
    time.sleep(3)
    
    # Track timing for breaks
    start_time = time.time()
    last_break_time = start_time
    prompt_count = 0
    
    print(f"\nStarting automation with {len(prompts)} prompts")
    print("Press Ctrl+C to stop at any time\n")
    
    try:
        for i, prompt in enumerate(prompts):
            # Check if it's night time (midnight to 7 AM)
            if is_night_time():
                print("It's night time (midnight-7AM). Taking a full break.")
                print("Waiting until 7 AM to continue...")
                while is_night_time():
                    time.sleep(300)  # Check every 5 minutes
                print("Good morning! Resuming automation.")
            
            # Check for hourly break
            if should_take_hourly_break(start_time, last_break_time):
                break_duration = random.randint(10*60, 20*60)  # 10-20 minutes in seconds
                print(f"\nTaking a break for {break_duration//60} minutes...")
                time.sleep(break_duration)
                last_break_time = time.time()
                print("Break finished. Resuming...\n")
            
            print(f"Processing prompt {i+1}/{len(prompts)}")
            
            # Clear any existing text in textbox
            clear_textbox()
            
            # Step 1: Paste the image (already in clipboard)
            print("  1. Pasting image...")
            paste_from_clipboard()
            
            # Step 2: Wait 2-4 seconds
            wait_time = get_random_wait(2, 4)
            print(f"  2. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            
            # Step 3: Copy prompt to clipboard and paste
            print(f"  3. Pasting prompt: {prompt[:50]}...")
            pyperclip.copy(prompt)
            time.sleep(0.5)
            paste_from_clipboard()
            
            # Step 4: Wait 2-5 seconds
            wait_time = get_random_wait(2, 5)
            print(f"  4. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            
            # Step 5: Hit enter
            print("  5. Pressing Enter...")
            pyautogui.press('enter')
            
            # Step 6: Wait 2-5 minutes for response
            wait_time = get_random_wait(2*60, 5*60)  # 2-5 minutes in seconds
            print(f"  6. Waiting {wait_time/60:.1f} minutes for response...")
            time.sleep(wait_time)
            
            # Step 7: Start new chat (except for last prompt)
            if i < len(prompts) - 1:
                print("  7. Starting new chat...")
                start_new_chat()
                
                # Wait 2-6 seconds for new chat to load
                wait_time = get_random_wait(2, 6)
                print(f"  8. Waiting {wait_time:.1f} seconds for new chat...")
                time.sleep(wait_time)
                
                # Copy image back to clipboard for next iteration
                copy_image_to_clipboard(image_file)
            
            prompt_count += 1
            print(f"Completed prompt {prompt_count}\n")
            
    except KeyboardInterrupt:
        print(f"\n\nAutomation stopped by user.")
        print(f"Processed {prompt_count} out of {len(prompts)} prompts.")
    except Exception as e:
        print(f"\n\nError during automation: {e}")
        print(f"Processed {prompt_count} out of {len(prompts)} prompts.")
    
    print("\nAutomation complete!")

if __name__ == "__main__":
    print("=== ChatGPT Image Creation Automation ===")
    print("Prerequisites:")
    print("1. Prompts file at: /Users/amirshachar/Desktop/image_prompts.txt")
    print("2. Image file at: ~/Downloads/man.png")
    print("\nMake sure Chrome is installed and you're logged into ChatGPT.")
    print("The script will handle the rest automatically.")
    
    input("\nPress Enter to start automation...")
    main()