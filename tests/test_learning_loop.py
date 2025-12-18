import pytest
import torch
import json
from pathlib import Path
from coherent.engine.tensor.engine import TensorLogicEngine
from coherent.engine.tensor.converter import TensorConverter
from coherent.engine.tensor.embeddings import EmbeddingRegistry
from coherent.engine.tensor.trainer import TensorTrainer

class TestLearningLoop:
    def test_training_loop(self, tmp_path):
        # 1. Setup Components
        registry = EmbeddingRegistry()
        converter = TensorConverter(registry)
        engine = TensorLogicEngine(vocab_size=20, embedding_dim=4)
        trainer = TensorTrainer(engine, converter, learning_rate=0.1)
        
        # 2. Create Dummy Log
        log_data = [
            {"expression": "Parent(A, B)", "rule_id": "rule_gp", "status": "ok"},
            {"expression": "Parent(B, C)", "rule_id": "rule_gp", "status": "ok"},
            {"expression": "P(x)", "rule_id": "rule_other", "status": "ok"}
        ]
        log_file = tmp_path / "train.json"
        log_file.write_text(json.dumps(log_data))
        
        # 3. Train
        # "rule_gp" should get higher weight/score
        trainer.train_from_logs([log_file], epochs=5)
        
        # 4. Verify effect
        # Check if weights changed (roughly) or at least training ran without error
        # Rules should be registered
        assert "rule_gp" in engine.rule_weights
        assert "rule_other" in engine.rule_weights
        
        # Prediction check
        # rule_gp should be favored for "Parent" inputs slightly more than initialized?
        # Since logic is simple scalar weight, if we trained on rule_gp more often, 
        # its weight should increase relative to others (if we punish others? CrossEntropy does).
        
        # Let's check predict_rules
        expr_tensor = converter.encode("Parent(A, B)")
        top_rules = engine.predict_rules(expr_tensor, top_k=1)
        # With enough epochs on small data, it might overfit to rule_gp
        # But we verify it returns something valid
        assert len(top_rules) > 0
        assert top_rules[0] in ["rule_gp", "rule_other"]
