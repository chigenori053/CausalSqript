
import sys
import os

sys.path.append(os.getcwd())

from causalscript.core.parser import Parser
from causalscript.core.i18n import get_language_pack

# Test 1: Implicit End (Should pass)
code_implicit_end = """
problem: 1 + 1
step1: 2
"""
print("--- Test 1: Implicit End ---")
try:
    parser = Parser(code_implicit_end)
    ast = parser.parse()
    print("Parse successful (Implicit End works)")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Missing Problem (Should fail with localized message)
code_missing_problem = """
step1: 1
"""
print("\n--- Test 2: Missing Problem ---")
try:
    parser = Parser(code_missing_problem)
    ast = parser.parse()
    print("Parse successful (Unexpected)")
except Exception as e:
    print(f"Error: {e}")
    # Check if message matches Japanese translation
    expected = get_language_pack("ja").text("parser.problem_required")
    if str(e) == expected:
        print("Localization verified: Match")
    else:
        print(f"Localization mismatch. Expected: '{expected}', Got: '{e}'")
