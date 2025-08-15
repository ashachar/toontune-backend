#!/usr/bin/env python3
"""
Replicate Balance Hooks
Tracks Replicate account balance before and after inference to calculate costs
"""

import os
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ReplicateBalanceTracker:
    def __init__(self):
        self.api_token = os.getenv('REPLICATE_API_TOKEN') or os.getenv('REPLICATE_API_KEY')
        self.headers = {
            'Authorization': f'Token {self.api_token}',
            'Content-Type': 'application/json'
        }
        self.balance_before = None
        self.balance_after = None
        self.start_time = None
        
    def get_balance(self):
        """
        Get current Replicate account balance
        NOTE: Replicate API doesn't expose credit balance, so we track predictions instead
        """
        # Unfortunately, Replicate doesn't expose credit balance via API
        # We can only track by counting predictions or manual entry
        return None  # Will be enhanced when Replicate adds this to their API
    
    def get_balance_manual(self):
        """Allow manual balance entry for tracking"""
        balance_file = Path(".replicate_balance")
        if balance_file.exists():
            try:
                return float(balance_file.read_text().strip())
            except:
                pass
        return None
    
    def set_balance_manual(self, balance):
        """Manually set the balance for tracking"""
        balance_file = Path(".replicate_balance")
        balance_file.write_text(str(balance))
        return balance
    
    def pre_inference_hook(self):
        """Hook to run before inference - captures initial balance"""
        print("\n" + "="*60)
        print("💰 REPLICATE BALANCE CHECK (PRE-INFERENCE)")
        print("="*60)
        
        self.start_time = datetime.now()
        self.balance_before = self.get_balance()
        
        if self.balance_before is not None:
            print(f"📊 Current Balance: ${self.balance_before:.4f}")
            
            # Warning if balance is low
            if self.balance_before < 1.0:
                print(f"⚠️  WARNING: Low balance! Only ${self.balance_before:.4f} remaining")
            elif self.balance_before < 5.0:
                print(f"📢 Note: Balance below $5")
        else:
            print("⚠️  Could not retrieve balance")
        
        print("="*60 + "\n")
        return self.balance_before
    
    def post_inference_hook(self, model_name=None, duration=None):
        """Hook to run after inference - calculates cost"""
        print("\n" + "="*60)
        print("💰 REPLICATE BALANCE CHECK (POST-INFERENCE)")
        print("="*60)
        
        self.balance_after = self.get_balance()
        
        if self.balance_after is not None:
            print(f"📊 Current Balance: ${self.balance_after:.4f}")
            
            if self.balance_before is not None:
                # Calculate cost
                cost = self.balance_before - self.balance_after
                
                if cost > 0:
                    print(f"\n💸 INFERENCE COST: ${cost:.4f}")
                    
                    # Additional details
                    if model_name:
                        print(f"   Model: {model_name}")
                    if duration:
                        print(f"   Duration: {duration} seconds")
                    
                    # Calculate time taken
                    if self.start_time:
                        time_taken = (datetime.now() - self.start_time).total_seconds()
                        print(f"   Time taken: {time_taken:.1f} seconds")
                    
                    # Cost per second if video duration provided
                    if duration and duration > 0:
                        cost_per_second = cost / duration
                        print(f"   Cost per second: ${cost_per_second:.4f}")
                    
                    # Warning if expensive
                    if cost > 1.0:
                        print(f"\n⚠️  High cost inference: ${cost:.2f}!")
                    
                    # Log to file
                    self.log_cost_to_file(model_name, duration, cost, time_taken)
                    
                elif cost == 0:
                    print("\n✅ No cost incurred (possibly cached or free tier)")
                else:
                    print(f"\n✨ Balance increased by ${abs(cost):.4f} (credits added?)")
            else:
                print("⚠️  Cannot calculate cost (no pre-inference balance)")
        else:
            print("⚠️  Could not retrieve post-inference balance")
        
        print("="*60 + "\n")
        return self.balance_after
    
    def log_cost_to_file(self, model_name, duration, cost, time_taken):
        """Log cost details to a file for tracking"""
        log_file = Path("replicate_costs.log")
        
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Duration: {duration}s\n")
            f.write(f"Cost: ${cost:.4f}\n")
            f.write(f"Time taken: {time_taken:.1f}s\n")
            f.write(f"Balance before: ${self.balance_before:.4f}\n")
            f.write(f"Balance after: ${self.balance_after:.4f}\n")
            f.write(f"{'='*60}\n")
    
    def get_cost_summary(self):
        """Get a summary of the cost"""
        if self.balance_before is not None and self.balance_after is not None:
            cost = self.balance_before - self.balance_after
            return {
                'balance_before': self.balance_before,
                'balance_after': self.balance_after,
                'cost': cost,
                'timestamp': datetime.now().isoformat()
            }
        return None

# Global tracker instance
_tracker = None

def get_tracker():
    """Get or create global tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = ReplicateBalanceTracker()
    return _tracker

def pre_inference_balance_check():
    """Standalone function for pre-inference hook"""
    tracker = get_tracker()
    return tracker.pre_inference_hook()

def post_inference_balance_check(model_name=None, duration=None):
    """Standalone function for post-inference hook"""
    tracker = get_tracker()
    return tracker.post_inference_hook(model_name, duration)

def reset_tracker():
    """Reset the global tracker"""
    global _tracker
    _tracker = None

if __name__ == "__main__":
    # Test the balance checker
    print("Testing Replicate Balance Tracker...")
    
    tracker = ReplicateBalanceTracker()
    
    # Test getting balance
    balance = tracker.get_balance()
    if balance is not None:
        print(f"✅ Successfully retrieved balance: ${balance:.4f}")
    else:
        print("❌ Could not retrieve balance")
    
    # Simulate pre/post hooks
    print("\nSimulating inference hooks...")
    tracker.pre_inference_hook()
    
    import time
    print("(Simulating inference delay...)")
    time.sleep(2)
    
    tracker.post_inference_hook(model_name="kwaivgi/kling-v2.1", duration=5)