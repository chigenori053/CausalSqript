from causalscript.core.parser import Parser
from causalscript.core.errors import SyntaxError

def test_reproduce_matrix_prepare_error():
    source = """
problem: Matrix Calculation
prepare:
  - A = [1, 2
         3, 4]
  - B = [2, 0
         1, 2]
"""
    # This currently fails with TokenError or similar because _parse_prepare_block 
    # doesn't handle multi-line lists cleanly.
    print("Testing Prepare Block Parsing...")
    try:
        parser = Parser(source)
        parser.parse()
        print("FAIL: Prepare block parsed successfully (unexpected).")
    except Exception as e:
        print(f"SUCCESS: Caught expected error in prepare: {e}")

def test_step_list_format_error():
    source = """
problem: Matrix Calc
step:
  - A = [5, 2]
  - B = [6, 4]
"""
    # Unless this triggers _is_mapping_block -> True, it might default to _parse_step_multiline 
    # which might work or not. But user says "Step block missing 'after'".
    
    print("\nTesting Step Block Parsing...")
    parser = Parser(source)
    try:
        nodes = parser.parse().body
        print(f"Parsed nodes: {nodes}")
    except Exception as e:
        print(f"Caught error in step parsing: {e}")

def test_matrix_operation_error():
    # Test if the engine can handle basic matrix multiplication syntax for lists
    # This requires both Parser support (already fixed) and Engine support (to be fixed)
    source = """
problem: Matrix Mult
prepare:
  - A = [1, 2
         3, 4]
  - B = [2, 0
         1, 2]
step:
  - C = A * B
"""
    print("\nTesting Matrix Operation Execution...")
    
    # 1. Parse
    parser = Parser(source)
    program = parser.parse()
    
    # 2. Simulate Engine behavior
    try:
        from causalscript.core.symbolic_engine import SymbolicEngine, _FallbackEvaluator
        
        fallback = _FallbackEvaluator()
        
        # Manually create lists (simulating parsed output)
        A = [[1, 2], [3, 4]]
        B = [[2, 0], [1, 2]]
        
        # Test A * B
        expr = "A * B"
        context = {"A": A, "B": B}
        result = fallback.evaluate(expr, context)
        print(f"Result of A * B: {result}")
        
    except Exception as e:
        print(f"Engine execution failed: {e}")

if __name__ == "__main__":
    test_reproduce_matrix_prepare_error()
    test_step_list_format_error()
    test_matrix_operation_error()
