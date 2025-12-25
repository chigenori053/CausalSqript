"""
Example script demonstrating COHERENT Library usage.
Corresponds to the Quick Start in LIBRARY_MANUAL.md.
"""
import sys
import logging
from coherent import CoreRuntime, StatusManager, SystemState

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    print("=== COHERENT Library Demo ===")
    
    # 1. Check Status
    manager = StatusManager()
    status = manager.get_status()
    print(f"[Status] ID: {status.system_id}, State: {status.mode.state}")
    
    if status.mode.state != SystemState.NORMAL:
        print("System is not in NORMAL state. Exiting.")
        sys.exit(1)

    # 2. Initialize Runtime
    print("[Init] Initializing CoreRuntime...")
    # Note: These internal imports are needed for construction, 
    # but the main entry is CoreRuntime (exported in top-level init).
    from coherent.core.computation_engine import ComputationEngine
    from coherent.core.validation_engine import ValidationEngine
    from coherent.core.hint_engine import HintEngine
    from coherent.core.symbolic_engine import SymbolicEngine

    try:
        symbolic = SymbolicEngine()
        comp_engine = ComputationEngine(symbolic)
        val_engine = ValidationEngine(comp_engine) 
        hint_engine = HintEngine(comp_engine)

        runtime = CoreRuntime(
            computation_engine=comp_engine,
            validation_engine=val_engine,
            hint_engine=hint_engine
        )
        print("[Init] Success.")
    except Exception as e:
        print(f"[Init] Failed: {e}")
        sys.exit(1)

    # 3. Validation Logic
    problem = "x**2 + 2*x + 1"
    step = "(x + 1)**2"
    
    print(f"\n[Validation] Problem: {problem}")
    print(f"[Validation] Step:    {step}")
    
    runtime.set(problem)
    result = runtime.check_step(step)
    
    print(f"[Result] Valid: {result['valid']}")
    print(f"[Result] Status: {result['details'].get('status')}")
    
    if result['valid']:
        print("✅ Validation successful!")
    else:
        print("❌ Validation failed.")

if __name__ == "__main__":
    main()
