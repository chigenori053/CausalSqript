# Edu Demo

`edu_demo_runner.py` loads scenarios from `edu/config/edu_demo_config.yaml` and invokes the Edu CLI.

- `basic_arithmetic` → `edu/examples/pythagorean.mlang`
- `counterfactual` → `edu/examples/counterfactual_demo.mlang`

Run:

```bash
python -m edu.demo.edu_demo_runner basic_arithmetic
python -m edu.demo.edu_demo_runner counterfactual --with-counterfactual
```
