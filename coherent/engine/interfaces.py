from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseParser(ABC):
    """Abstract base class for domain-specific parsers."""
    
    @abstractmethod
    def parse(self, text: str) -> Any:
        """
        Parse the text into an AST or internal representation.
        Raises SyntaxError or specialized error on failure.
        """
        pass

    @abstractmethod
    def validate(self, text: str) -> bool:
        """
        Check if the text is valid for this domain.
        Returns True if valid, False otherwise.
        """
        pass

class BaseEngine(ABC):
    """Abstract base class for domain-specific computation engines."""
    
    @abstractmethod
    def evaluate(self, node: Any, context: Dict[str, Any]) -> Any:
        """
        Evaluate the AST node within the given context.
        """
        pass

class BaseModule(ABC):
    """Abstract base class for a domain module."""
    parser: BaseParser
    engine: BaseEngine
