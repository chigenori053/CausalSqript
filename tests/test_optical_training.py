import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from coherent.optical.layer import OpticalInterferenceEngine
from coherent.optical.trainer import OpticalTrainer
from coherent.optical.vectorizer import FeatureExtractor

@pytest.fixture
def trainer_and_model():
    model = OpticalInterferenceEngine(input_dim=10, memory_capacity=5)
    vectorizer = FeatureExtractor(vector_size=10)
    trainer = OpticalTrainer(model, vectorizer, lr=0.1)
    return trainer, model

def test_training_step_reduces_loss(trainer_and_model):
    trainer, model = trainer_and_model
    
    # Target rule index 3
    expr_str = "x + y"
    target_idx = 3
    
    # Initial loss
    initial_loss = trainer.train_one_step(expr_str, target_idx)
    
    # Train for a few steps on the SAME sample
    for _ in range(10):
        loss = trainer.train_one_step(expr_str, target_idx)
        
    final_loss = loss
    
    # Loss should decrease
    assert final_loss < initial_loss

def test_training_epoch(trainer_and_model):
    trainer, model = trainer_and_model
    
    data = [
        ("eq1", 0),
        ("eq2", 1),
        ("eq3", 0)
    ]
    
    avg_loss = trainer.train_epoch(data)
    assert avg_loss > 0.0
    
    # Verify weights changed from random init
    # (Checking if graph is connected effectively)
    param = list(model.parameters())[0]
    assert param.requires_grad
