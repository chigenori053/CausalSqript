import pytest
import json
import yaml
from pathlib import Path
from coherent.engine.exercise_spec import ExerciseSpec, load_exercise_spec, save_exercise_spec

@pytest.fixture
def sample_spec():
    return ExerciseSpec(
        id="test_ex",
        target_expression="x**2",
        validation_mode="symbolic_equiv",
        hint_rules={"x": "Too simple"},
        metadata={"difficulty": "easy"}
    )

def test_to_dict(sample_spec):
    data = sample_spec.to_dict()
    assert data["id"] == "test_ex"
    assert data["target_expression"] == "x**2"
    assert data["hint_rules"] == {"x": "Too simple"}

def test_from_dict(sample_spec):
    data = sample_spec.to_dict()
    spec = ExerciseSpec.from_dict(data)
    assert spec == sample_spec

def test_json_io(sample_spec, tmp_path):
    path = tmp_path / "test.json"
    save_exercise_spec(sample_spec, path)
    
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
        assert data["id"] == "test_ex"
        
    loaded_spec = load_exercise_spec(path)
    assert loaded_spec == sample_spec

def test_yaml_io(sample_spec, tmp_path):
    path = tmp_path / "test.yaml"
    save_exercise_spec(sample_spec, path)
    
    assert path.exists()
    with open(path) as f:
        data = yaml.safe_load(f)
        assert data["id"] == "test_ex"
        
    loaded_spec = load_exercise_spec(path)
    assert loaded_spec == sample_spec

def test_unsupported_format(sample_spec, tmp_path):
    path = tmp_path / "test.txt"
    with pytest.raises(ValueError, match="Unsupported file format"):
        save_exercise_spec(sample_spec, path)
        
    path.touch()
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_exercise_spec(path)

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_exercise_spec("nonexistent.json")
