import unittest
from coherent.engine.parser import Parser
from coherent.engine.errors import SyntaxError

class TestParserTypos(unittest.TestCase):
    def assert_suggestion(self, script, expected_suggestion):
        with self.assertRaises(SyntaxError) as cm:
            parser = Parser(script)
            parser.parse()
        self.assertIn(f"Did you mean '{expected_suggestion}'?", str(cm.exception))

    def test_problem_typo(self):
        self.assert_suggestion("roblem: x", "problem")
        self.assert_suggestion("prblem: x", "problem")

    def test_step_typo(self):
        script = """
        problem: x
        ste: x
        """
        self.assert_suggestion(script, "step")

    def test_end_typo(self):
        script = """
        problem: x
        step: x
        en: done
        """
        self.assert_suggestion(script, "end")

    def test_explain_typo(self):
        script = """
        problem: x
        explai: "foo"
        """
        self.assert_suggestion(script, "explain")

    def test_meta_typo(self):
        script = """
        problem: x
        met:
          key: value
        """
        self.assert_suggestion(script, "meta")

    def test_sub_problem_typo(self):
        script = """
        problem: x
        sub_problm: y
        """
        self.assert_suggestion(script, "sub_problem")

    def test_no_suggestion_for_garbage(self):
        script = """
        problem: x
        asdfghjkl: y
        """
        with self.assertRaises(SyntaxError) as cm:
            parser = Parser(script)
            parser.parse()
        self.assertNotIn("Did you mean", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
