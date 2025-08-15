#!/usr/bin/env python3
"""
Replicate Cost Tracker
Since Replicate API doesn't expose balance, we track costs based on model pricing
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ReplicateCostTracker:
    """Track Replicate costs based on known model pricing"""
    
    # Known model costs (approximate, in USD)
    # Based on Replicate's typical pricing structure
    MODEL_COSTS = {
        'kwaivgi/kling-v2.1': {
            'standard': 0.025,  # $0.025 per second for 720p
            'pro': 0.050,       # $0.050 per second for 1080p
        },
        'kwaivgi/kling-v2.1-master': {
            'default': 0.035,   # Average cost
        },
        'fofr/tooncrafter': {
            'default': 0.020,   # $0.02 per second
        }
    }
    
    def __init__(self):
        self.cost_log_file = Path("replicate_costs.json")
        self.balance_file = Path(".replicate_balance")
        self.load_balance()
        self.start_time = None
        
    def load_balance(self):
        """Load manually set balance"""
        if self.balance_file.exists():
            try:
                self.current_balance = float(self.balance_file.read_text().strip())
            except:
                self.current_balance = 10.0  # Default to your $10 credit
        else:
            self.current_balance = 10.0  # You have $10 credit
            self.save_balance()
    
    def save_balance(self):
        """Save current balance"""
        self.balance_file.write_text(f"{self.current_balance:.4f}")
    
    def estimate_cost(self, model_name, duration=5, mode='standard'):
        """Estimate cost for a model run"""
        # Get base model name without version
        base_model = model_name.split(':')[0] if ':' in model_name else model_name
        
        if base_model in self.MODEL_COSTS:
            costs = self.MODEL_COSTS[base_model]
            cost_per_second = costs.get(mode, costs.get('default', 0.030))
        else:
            # Default estimate for unknown models
            cost_per_second = 0.030
        
        total_cost = cost_per_second * duration
        return total_cost, cost_per_second
    
    def pre_inference_hook(self, model_name=None, duration=5, mode='standard'):
        """Hook before running inference"""
        self.start_time = datetime.now()
        
        print("\n" + "="*60)
        print("💰 REPLICATE COST TRACKER (PRE-INFERENCE)")
        print("="*60)
        print(f"📊 Current Balance: ${self.current_balance:.2f}")
        
        if model_name:
            estimated_cost, cost_per_sec = self.estimate_cost(model_name, duration, mode)
            print(f"📈 Estimated Cost: ${estimated_cost:.4f}")
            print(f"   Model: {model_name}")
            print(f"   Mode: {mode} ({'720p' if mode == 'standard' else '1080p' if mode == 'pro' else 'default'})")
            print(f"   Duration: {duration} seconds")
            print(f"   Rate: ${cost_per_sec:.4f}/second")
            
            if estimated_cost > self.current_balance:
                print(f"\n⚠️  WARNING: Estimated cost (${estimated_cost:.2f}) exceeds balance (${self.current_balance:.2f})!")
            elif self.current_balance - estimated_cost < 1.0:
                print(f"\n📢 Note: Balance will be low after this run (${self.current_balance - estimated_cost:.2f} remaining)")
        
        print("="*60 + "\n")
        return self.current_balance
    
    def post_inference_hook(self, model_name, duration=5, mode='standard', actual_cost=None):
        """Hook after inference completes"""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        print("\n" + "="*60)
        print("💰 REPLICATE COST TRACKER (POST-INFERENCE)")
        print("="*60)
        
        # Calculate cost
        if actual_cost is not None:
            # If we somehow know the actual cost
            cost = actual_cost
            cost_per_sec = cost / duration if duration > 0 else 0
        else:
            # Use our estimation
            cost, cost_per_sec = self.estimate_cost(model_name, duration, mode)
        
        # Update balance
        old_balance = self.current_balance
        self.current_balance -= cost
        self.save_balance()
        
        print(f"📊 Previous Balance: ${old_balance:.2f}")
        print(f"💸 Inference Cost: ${cost:.4f}")
        print(f"📊 New Balance: ${self.current_balance:.2f}")
        print(f"\n📝 Details:")
        print(f"   Model: {model_name}")
        print(f"   Mode: {mode}")
        print(f"   Duration: {duration} seconds")
        print(f"   Rate: ${cost_per_sec:.4f}/second")
        print(f"   Processing time: {elapsed:.1f} seconds")
        
        # Warnings
        if self.current_balance < 0:
            print(f"\n❌ BALANCE DEPLETED! You're over by ${abs(self.current_balance):.2f}")
        elif self.current_balance < 1.0:
            print(f"\n⚠️  LOW BALANCE: Only ${self.current_balance:.2f} remaining!")
        elif self.current_balance < 5.0:
            print(f"\n📢 Balance below $5: ${self.current_balance:.2f} remaining")
        
        # Log to file
        self.log_cost(model_name, mode, duration, cost, elapsed)
        
        print("="*60 + "\n")
        return self.current_balance
    
    def log_cost(self, model_name, mode, duration, cost, elapsed):
        """Log cost details to JSON file"""
        # Load existing log
        if self.cost_log_file.exists():
            try:
                with open(self.cost_log_file) as f:
                    log = json.load(f)
            except:
                log = []
        else:
            log = []
        
        # Add new entry
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'mode': mode,
            'duration_seconds': duration,
            'cost_usd': round(cost, 4),
            'processing_time_seconds': round(elapsed, 1),
            'balance_after': round(self.current_balance, 2)
        }
        log.append(entry)
        
        # Save
        with open(self.cost_log_file, 'w') as f:
            json.dump(log, f, indent=2)
        
        print(f"📁 Cost logged to: {self.cost_log_file}")
    
    def get_total_spent(self):
        """Get total amount spent from logs"""
        if not self.cost_log_file.exists():
            return 0
        
        try:
            with open(self.cost_log_file) as f:
                log = json.load(f)
            return sum(entry.get('cost_usd', 0) for entry in log)
        except:
            return 0
    
    def reset_balance(self, new_balance=10.0):
        """Reset balance to a new value"""
        self.current_balance = new_balance
        self.save_balance()
        print(f"✅ Balance reset to ${new_balance:.2f}")

# Global instance
_tracker = None

def get_cost_tracker():
    """Get or create global tracker"""
    global _tracker
    if _tracker is None:
        _tracker = ReplicateCostTracker()
    return _tracker

def track_cost_pre(model_name, duration=5, mode='standard'):
    """Pre-inference cost tracking"""
    tracker = get_cost_tracker()
    return tracker.pre_inference_hook(model_name, duration, mode)

def track_cost_post(model_name, duration=5, mode='standard'):
    """Post-inference cost tracking"""
    tracker = get_cost_tracker()
    return tracker.post_inference_hook(model_name, duration, mode)

if __name__ == "__main__":
    # Test the cost tracker
    print("Testing Replicate Cost Tracker...")
    print("="*50)
    
    tracker = ReplicateCostTracker()
    
    # Show current balance
    print(f"Current balance: ${tracker.current_balance:.2f}")
    print(f"Total spent so far: ${tracker.get_total_spent():.4f}")
    
    # Simulate a Kling run
    print("\nSimulating Kling v2.1 standard mode (5 seconds)...")
    tracker.pre_inference_hook('kwaivgi/kling-v2.1', duration=5, mode='standard')
    
    import time
    time.sleep(2)  # Simulate processing
    
    tracker.post_inference_hook('kwaivgi/kling-v2.1', duration=5, mode='standard')
    
    print(f"\nNew balance: ${tracker.current_balance:.2f}")