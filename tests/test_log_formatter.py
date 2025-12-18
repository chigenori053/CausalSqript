from coherent.engine.log_formatter import format_record_message, format_records


def test_format_record_message_includes_meta_lines():
    record = {
        "phase": "step",
        "step_index": 2,
        "expression": "x**2 - x*y + y**2",
        "status": "mistake",
        "meta": {"reason": "invalid_step", "explanation": "Compared via sympy"},
    }
    lines = format_record_message(record)
    assert lines[0] == "step2: x**2 - x*y + y**2"
    assert "message: invalid_step" in lines
    assert "explain: Compared via sympy" in lines


def test_format_records_handles_problem_and_end():
    records = [
        {"phase": "problem", "expression": "(x - y)**2", "status": "ok", "meta": {}},
        {"phase": "step", "step_index": 1, "expression": "(x - y)*(x - y)", "status": "ok", "meta": {}},
        {"phase": "end", "rendered": "End: done", "status": "ok", "meta": {}},
    ]
    lines = format_records(records)
    assert lines[0] == "problem: (x - y)**2"
    assert "step1: (x - y)*(x - y)" in lines
    assert "end: done" in lines
