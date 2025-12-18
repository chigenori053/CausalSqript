
import unittest
from coherent.engine.parser import Parser

class TestMatrixParsing(unittest.TestCase):
    def test_multiline_matrix(self):
        source = """
problem:
  行列A = [ 1, 0, 1
          2, 3, 4
          5, 6, 7 ]
step: 行列A
end: done
"""
        parser = Parser(source)
        try:
            program = parser.parse()
            print("Parsed Nodes:", program.body)
            # Check what we got
            # We expect a ProblemNode containing... actually just text?
            # Or StepNode with correct expression?
            # The example puts it in 'problem', effectively an assignment/condition.
            # parser.py _parse_problem_block uses similar logic to step blocks usually?
            # actually _parse_problem just takes content. 
            # If it's a block: _collect_block -> _parse_problem_block.
            
            # Let's see node structure.
            for node in program.body:
                print(f"Node: {type(node)} -> {vars(node)}")
                
        except Exception as e:
            print(f"Parsing Failed: {e}")
            raise e

if __name__ == "__main__":
    unittest.main()
