from typing import List
from .math_category import MathCategory, CategoryResult
from .symbolic_engine import SymbolicEngine
from .classifier import ExpressionClassifier

class CategoryIdentifier:
    """
    Identifies the mathematical category of a given expression.
    Wraps the lower-level ExpressionClassifier to provide structured output.
    """

    def __init__(self, engine: SymbolicEngine):
        self.engine = engine
        self.classifier = ExpressionClassifier(engine)

    def identify(self, expr: str) -> CategoryResult:
        """
        Identifies the category of the expression.
        
        Args:
            expr: The mathematical expression string.
            
        Returns:
            CategoryResult containing the primary category and related info.
        """
        # Use the existing classifier to get domain strings
        domain_strings = self.classifier.classify(expr)
        
        if not domain_strings:
            return CategoryResult(primary_category=MathCategory.UNKNOWN, confidence=0.0)

        # Map strings to Enums
        categories = []
        for d in domain_strings:
            try:
                categories.append(MathCategory(d))
            except ValueError:
                # Handle unknown domains gracefully
                pass
        
        if not categories:
             return CategoryResult(primary_category=MathCategory.UNKNOWN, confidence=0.0)

        # The first one is the primary category based on ExpressionClassifier's priority
        primary = categories[0]
        related = categories[1:]
        
        return CategoryResult(
            primary_category=primary,
            confidence=1.0, # Placeholder confidence
            related_categories=related,
            details={"raw_domains": domain_strings}
        )
