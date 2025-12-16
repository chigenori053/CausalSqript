
import sys
import pytest

# Redirect stdout/stderr to a file
with open("test_output.log", "w") as f:
    sys.stdout = f
    sys.stderr = f
    
    print("Running tests...")
    retcode = pytest.main(["tests/knowledge/test_complex_knowledge.py"])
    print(f"Pytest finished with code: {retcode}")
