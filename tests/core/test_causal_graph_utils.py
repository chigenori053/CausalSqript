from coherent.logic.graph_utils import graph_to_dot, graph_to_text


def _sample_graph():
    return {
        "nodes": [
            {"id": "problem-1", "type": "problem", "payload": {"record": {"rendered": "Problem"}}},
            {"id": "step-1", "type": "step", "payload": {"record": {"rendered": "Step"}}},
        ],
        "edges": [
            {"source": "problem-1", "target": "step-1", "type": "step_transition"},
        ],
    }


def test_graph_to_text_includes_nodes_and_edges():
    output = graph_to_text(_sample_graph())
    assert "problem-1" in output
    assert "step-1" in output
    assert "step_transition" in output


def test_graph_to_dot_builds_valid_string():
    dot = graph_to_dot(_sample_graph())
    assert "digraph" in dot
    assert '"problem-1"' in dot
    assert "step_transition" in dot
