
import unittest
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine

class TestSystemImplication(unittest.TestCase):
    def test_implication_check(self):
        sym_engine = SymbolicEngine()
        comp_engine = ComputationEngine(sym_engine)
        val_engine = ValidationEngine(comp_engine)
        hint_engine = HintEngine(comp_engine)
        runtime = CoreRuntime(comp_engine, val_engine, hint_engine)

        # Setup problem: System of equations
        # 3x + 2y = 14
        # x - 2y = 2
        runtime.set("System(Eq(3*x + 2*y, 14), Eq(x - 2*y, 2))")
        
        # Test step: x = 4 (Implied by the system)
        # This should now be valid (Partial/Implication)
        result = runtime.check_step("x = 4")
        
        print(f"Check Step Result: valid={result['valid']}, status={result['details'].get('status')}")
        
        self.assertTrue(result['valid'], "x=4 should be valid as it is implied by the system")
        self.assertEqual(result['details'].get('status'), "partial", "Status should be partial/implied")
        self.assertTrue(result['details'].get('partial'), "Should be marked as partial")

if __name__ == "__main__":
    unittest.main()
