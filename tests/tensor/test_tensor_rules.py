import pytest
import torch
from causalscript.core.tensor.layers import TensorRuleLayer, LogicChainLayer, LogicAndLayer
from causalscript.core.tensor.functional import prob_and, prob_or, prob_not

class TestTensorRules:
    def test_chain_rule_logic(self):
        # Transitive Inference: A -> B, B -> C  =>  A -> C
        # Let's simulate:
        # P(x, y) matrix where x is Parent of y
        
        # Batch size = 1
        # Entities: 0=Grandpa, 1=Dad, 2=Kid
        # Parent(Grandpa, Dad) = 1.0 (at 0,1)
        # Parent(Dad, Kid) = 1.0 (at 1,2)
        
        parent_rel = torch.zeros(1, 3, 3)
        parent_rel[0, 0, 1] = 1.0
        parent_rel[0, 1, 2] = 1.0
        
        # Apply Chain Rule: Parent(x,y) ^ Parent(y,z) -> GrandParent(x,z)
        layer = LogicChainLayer()
        # Note: Input usually expects probability logits if sigmoid is used internally?
        # But our layer applies sigmoid at end. If inputs are already 0/1 probabilities, 
        # the einsum sum will be 1.0. Sigmoid(1.0) is ~0.73, which is not 1.0.
        # Ideally, TensorRuleLayer should take logits or we shouldn't use Sigmoid if inputs are probs.
        # Let's check implementation. The layer applies sigmoid.
        # So inputs should act like logits or factors contributing to logits.
        
        # If we treat inputs as probabilities, we might not want sigmoid at output for pure logical deduction,
        # unless it's a "soft" deduction layer.
        # For this test, let's assume we want to verify the structural contraction correct.
        
        # Let's use a raw contraction for precise check
        raw_layer = TensorRuleLayer('bxy,byz->bxz', activation='none')
        
        grandparent_rel = raw_layer(parent_rel, parent_rel)
        
        # Expect GP(0, 2) to be 1.0 * 1.0 = 1.0 (from 0->1 * 1->2)
        assert grandparent_rel[0, 0, 2] == 1.0
        # Expect GP(0, 1) to be 0 (no path 0->?->1)
        assert grandparent_rel[0, 0, 1] == 0.0

    def test_functional_logic(self):
        t1 = torch.tensor([0.4, 0.8])
        t2 = torch.tensor([0.5, 0.2])
        
        # AND: .4*.5=.2, .8*.2=.16
        res_and = prob_and(t1, t2)
        assert torch.allclose(res_and, torch.tensor([0.2, 0.16]))
        
        # OR: .4+.5-.2=.7, .8+.2-.16=.84
        res_or = prob_or(t1, t2)
        assert torch.allclose(res_or, torch.tensor([0.7, 0.84]))
        
        # NOT: 1-.4=.6, 1-.8=.2
        res_not = prob_not(t1)
        assert torch.allclose(res_not, torch.tensor([0.6, 0.2]))

    def test_logic_layers_gradient(self):
        # Verify gradients flow
        a = torch.randn(1, 3, 3, requires_grad=True)
        b = torch.randn(1, 3, 3, requires_grad=True)
        
        layer = LogicChainLayer()
        output = layer(a, b)
        
        loss = output.sum()
        loss.backward()
        
        assert a.grad is not None
        assert b.grad is not None
