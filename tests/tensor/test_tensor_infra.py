import pytest
import torch
from coherent.engine.tensor.embeddings import EmbeddingRegistry
from coherent.engine.tensor.converter import TensorConverter
from coherent.engine.tensor.engine import TensorLogicEngine

class TestTensorInfrastructure:
    def test_embedding_registry(self):
        registry = EmbeddingRegistry()
        
        # Test 1: New registration
        id_a = registry.get_id("Parent")
        assert id_a > 0
        assert registry.get_term(id_a) == "Parent"
        
        # Test 2: Persistence
        id_b = registry.get_id("Parent")
        assert id_a == id_b
        
        # Test 3: Multiple unique terms
        id_c = registry.get_id("Child")
        assert id_a != id_c
        
        # Test 4: Vocab size (initial next_id=1, added 2 items -> next_id=3)
        assert registry.vocab_size == 3

    def test_tensor_converter(self):
        registry = EmbeddingRegistry()
        converter = TensorConverter(registry)
        
        expr = "Parent(x, y)"
        # Expected tokens: "Parent", "(", "x", ",", "y", ")"
        tokens = converter.tokenize(expr)
        assert tokens == ["Parent", "(", "x", ",", "y", ")"]
        
        # Encode
        tensor = converter.encode(expr)
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor) == 6
        assert tensor.dtype == torch.long
        
        # Check IDs are consistent
        idx_parent = registry.get_id("Parent")
        assert tensor[0] == idx_parent
        
        # Decode
        decoded = converter.decode(tensor)
        # Note: decode simply joins, so it might check consistency
        assert decoded == "Parent(x,y)"

    def test_batch_encoding(self):
        registry = EmbeddingRegistry()
        converter = TensorConverter(registry)
        
        exprs = ["A(x)", "B(y, z)"]
        batch = converter.batch_encode(exprs)
        
        # A(x) -> 4 tokens: A, (, x, )
        # B(y, z) -> 6 tokens: B, (, y, ,, z, )
        assert batch.shape == (2, 6)
        # Check padding (0 is default padding)
        assert batch[0, 4] == 0
        assert batch[0, 5] == 0

    def test_tensor_logic_engine_init(self):
        registry = EmbeddingRegistry()
        tensor = TensorLogicEngine(vocab_size=10, embedding_dim=16)
        
        # Check embeddings exist
        assert tensor.embeddings.weight.shape == (10, 16)
        
        # Check similarity calculation
        # Same index should have similarity 1.0 (approx)
        sim = tensor.get_similarity(1, 1)
        assert 0.99 < sim <= 1.0001
        
        # Different indices should compute
        sim_diff = tensor.get_similarity(1, 2)
        assert isinstance(sim_diff, float)

    def test_tensor_logic_engine_update(self):
        tensor = TensorLogicEngine(vocab_size=5, embedding_dim=8)
        old_weight = tensor.embeddings.weight.clone()
        
        # Expand vocab
        tensor.update_embeddings(vocab_size=10)
        assert tensor.embeddings.weight.shape == (10, 8)
        
        # Check old weights are preserved
        assert torch.allclose(tensor.embeddings.weight[:5], old_weight)
        # Check new weights are initialized (non-zero usually, or zero if default)
        # PyTorch embedding initializes to random normal usually
