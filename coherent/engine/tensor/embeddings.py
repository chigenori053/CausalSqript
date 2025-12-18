from typing import Dict, List, Optional

class EmbeddingRegistry:
    """
    Manages the mapping between symbolic terms (strings) and integer IDs.
    Used to tokenize concepts, variables, and rules for the TensorLogicEngine.
    """
    def __init__(self):
        self.term_to_id: Dict[str, int] = {}
        self.id_to_term: Dict[int, str] = {}
        # Start IDs from 1, reserving 0 for padding/null if needed
        self.next_id = 1

    def get_id(self, term: str, register_if_missing: bool = True) -> int:
        """
        Get the ID for a given term.
        If the term is not found and register_if_missing is True, assigns a new ID.
        Returns 0 (or raises) if missing and register_if_missing is False.
        """
        if term in self.term_to_id:
            return self.term_to_id[term]
        
        if register_if_missing:
            new_id = self.next_id
            self.term_to_id[term] = new_id
            self.id_to_term[new_id] = term
            self.next_id += 1
            return new_id
        
        # Default behavior for unknown terms when not registering: return 0 (UNK) mechanism?
        # For now, let's assume valid terms. Or return 0.
        return 0

    def get_term(self, idx: int) -> Optional[str]:
        """
        Get the term for a given ID. Returns None if ID is not found.
        """
        return self.id_to_term.get(idx)

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary (number of registered terms + 1 for safety).
        """
        return self.next_id
