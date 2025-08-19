"""
Step 5: LLM Inference
=====================

Runs LLM inference for generating effects based on prompts.
"""


class InferenceStep:
    """Handles LLM inference operations."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
    
    def run(self):
        """Run LLM inference (not in dry mode)."""
        print("\n" + "-"*60)
        print("STEP 5: RUNNING LLM INFERENCE")
        print("-"*60)
        
        if self.config.dry_run:
            print("  [DRY RUN MODE - Skipping LLM inference]")
            print("  No inference files will be created in dry-run mode")
        else:
            print("  [FULL MODE - Would call Gemini Pro here]")
            print("  ⚠ LLM inference not implemented in this version")
            print("  Please implement Gemini Pro integration")
            
            # TODO: Implement actual LLM inference here
            # 1. Load prompts from prompts directory
            # 2. Send to Gemini Pro API
            # 3. Save results to inferences directory
        
        self.pipeline_state['steps_completed'].append('run_inference')