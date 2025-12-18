from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from causalscript.core.memory.vector_store import VectorStoreBase
from causalscript.core.optical.layer import OpticalInterferenceEngine
from causalscript.core.holographic.data_types import HolographicTensor

class OpticalFrequencyStore(VectorStoreBase):
    """
    Optical Holographic Memory Store.
    
    Uses OpticalInterferenceEngine to store and recall vectors based on 
    phase resonance patterns rather than Euclidean distance.
    IMplements the VectorStoreBase interface.
    """
    
    def __init__(self, vector_dim: int = 384, capacity: int = 10000):
        self.vector_dim = vector_dim
        self.capacity = capacity
        
        # Initialize Optical Backend
        # Note: OpticalInterferenceEngine weights are [MemoryCapacity, InputDim]
        self.optical_layer = OpticalInterferenceEngine(
            input_dim=vector_dim, 
            memory_capacity=capacity
        )
        
        # Zero out the memory initially (Start with a blank slate/dark hologram)
        # Random initialization is better for "distributed" neural networks, 
        # but for a Key-Value store, we want empty slots to be empty.
        with torch.no_grad():
            self.optical_layer.optical_memory.data.fill_(0)
        
        # Metadata storage (Index -> ID/Meta)
        # In a real physical implementation, this would be holographically encoded too,
        # but for V1 we keep metadata in a digital lookup table.
        self.index_to_id: Dict[int, str] = {}
        self.index_to_metadata: Dict[int, Dict[str, Any]] = {}
        self.current_count = 0

    def _encode_signal(self, vectors: List[List[float]]) -> torch.Tensor:
        """
        Modulates real-valued vectors into complex holographic signals.
        """
        tensor = torch.tensor(vectors, dtype=torch.float32)
        
        # Normalize amplitude to unit sphere to ensure stable interference
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
        
        # Phase Encoding: simple amplitude embedding into complex plane for V1.
        # Future: Use phase-encoding exp(i * vector) for better capacity.
        return tensor.type(torch.cfloat)

    def add(self, collection_name: str, vectors: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """
        Write vectors to the optical memory (Holographic Exposure).
        """
        batch_size = len(vectors)
        if self.current_count + batch_size > self.capacity:
            raise MemoryError(f"Optical memory capacity exceeded ({self.capacity}).")

        # 1. Modulate signals
        signal = self._encode_signal(vectors)
        
        # 2. Write to Optical Layer (Flash Memory)
        with torch.no_grad():
            start = self.current_count
            end = start + batch_size
            
            # optical_memory is [Capacity, InputDim]
            # We copy the signals directly into the memory slots
            self.optical_layer.optical_memory.data[start:end] = signal

        # 3. Store Metadata
        for i, (meta, pid) in enumerate(zip(metadatas, ids)):
            idx = self.current_count + i
            self.index_to_id[idx] = pid
            self.index_to_metadata[idx] = meta or {}
            
        self.current_count += batch_size

    def query(self, collection_name: str, query_vec: List[float], filter: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recall memories via Optical Resonance.
        """
        # 1. Modulate Query
        input_tensor = self._encode_signal([query_vec]) # [1, Dim]
        
        # 2. Optical Interference (Forward Pass)
        # Returns resonance intensity [1, Capacity]
        intensity = self.optical_layer(input_tensor)
        
        # Calculate system ambiguity (uncertainty)
        ambiguity = self.optical_layer.get_ambiguity(intensity)
        
        # 3. Detect Signals
        scores = intensity[0].detach().cpu().numpy() # [Capacity]
        
        # Only consider currently utilized memory cells
        valid_scores = scores[:self.current_count]
        
        if len(valid_scores) == 0:
            return []
            
        # 4. Top-K Extraction
        # Argsort is ascending, so reverse it
        top_k = min(top_k, len(valid_scores))
        top_indices = np.argsort(valid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # Metadata Filter
            meta = self.index_to_metadata.get(idx, {})
            if filter:
                # Simple subset match
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue

            # Convert resonance score (energy) to distance metric
            # High Energy (Resonance) = Low Distance
            # Energy is roughly [0, 1] if vectors are normalized
            score = float(valid_scores[idx])
            distance = 1.0 - min(score, 1.0)
            
            results.append({
                "id": self.index_to_id.get(idx),
                "metadata": meta,
                "distance": distance,
                "score": score,
                "ambiguity": ambiguity # System-wide uncertainty for this query
            })
            
        return results

    def delete(self, collection_name: str, ids: List[str]) -> None:
        """
        Delete memories.
        In a physical optical memory, this might mean 'masking' or 'destructive interference'.
        Here we zero out the amplitude.
        """
        # Inefficient O(N) lookup for V1, but simpler than maintaining reverse index
        ids_to_delete = set(ids)
        
        indices_to_clear = []
        for idx, stored_id in self.index_to_id.items():
            if stored_id in ids_to_delete:
                indices_to_clear.append(idx)
        
        with torch.no_grad():
            for idx in indices_to_clear:
                # Zero out the physical memory
                self.optical_layer.optical_memory.data[idx] = torch.zeros(self.vector_dim, dtype=torch.cfloat)
                # Remove metadata ref
                if idx in self.index_to_id:
                    del self.index_to_id[idx]
                if idx in self.index_to_metadata:
                    del self.index_to_metadata[idx]
