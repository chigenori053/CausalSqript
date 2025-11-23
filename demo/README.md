# MathLang Core Extended Demos

This directory contains demonstrations of the Core Extended features in MathLang.

## Features Demonstrated

- **ComputationEngine**: Symbolic manipulation (simplify, expand, factor) and numeric evaluation
- **ValidationEngine**: Answer validation with multiple modes (symbolic equivalence, exact form, canonical form)
- **HintEngine**: Intelligent hint generation based on error patterns and heuristics
- **CoreRuntime**: Orchestrated problem-solving workflow integrating all engines

## Files

### CLI Demo
- `core_extended_demo.py` - Command-line demonstration script

**Usage:**
```bash
cd /Users/chigenori/development/mathlang
PYTHONPATH=. python demo/core_extended_demo.py
```

### Jupyter Notebook
- `core_extended_demo.ipynb` - Interactive notebook with explanations

**Usage:**
```bash
cd /Users/chigenori/development/mathlang/demo
jupyter notebook core_extended_demo.ipynb
```

### Sample Exercises
- `exercises/quadratic_expansion.yaml` - Exercise for expanding (x+1)^2
- `exercises/difference_of_squares.yaml` - Exercise for factoring x^2-9

## Exercise File Format

Exercises are defined in YAML format:

```yaml
id: "exercise_id"
target_expression: "correct_answer"
validation_mode: "symbolic_equiv"  # or "exact_form", "canonical_form"
canonical_form: null  # required for canonical_form mode
intermediate_steps:  # optional
  - "step1"
  - "step2"
hint_rules:  # optional
  "wrong_pattern": "Helpful hint message"
metadata:  # optional
  difficulty: "easy"
  topic: "algebra"
```

## Next Steps

- Try modifying the sample exercises
- Create your own exercises in YAML
- Experiment with different validation modes
- Add custom hint patterns
