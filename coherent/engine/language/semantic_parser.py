import re
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict

from coherent.engine.language.semantic_types import (
    SemanticIR, TaskType, MathDomain, GoalType, InputItem, 
    Constraints, LanguageMeta
)
from coherent.engine.input_parser import CoherentInputParser

class ISemanticParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> SemanticIR:
        pass

class RuleBasedSemanticParser(ISemanticParser):
    """
    A rule-based parser that uses Regex patterns to extract intent and domain.
    Supports English and Japanese.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define Keyword maps
        self._intent_patterns = {
            TaskType.SOLVE: [
                r"solve", r"calculate", r"compute", r"evaluate", r"derive", r"integrate",
                r"解く", r"解いて", r"計算", r"求めよ", r"微分", r"積分", r"計算して"
            ],
            TaskType.VERIFY: [
                r"verify", r"check", r"validate", r"confirm", r"is .* correct",
                r"確認", r"検証", r"正しい", r"合っているか", r"チェック"
            ],
            TaskType.HINT: [
                 r"hint", r"clue", r"help", r"stuck",
                 r"ヒント", r"助言", r"手助け", r"教えて", r"ここからどうすれば"
            ],
            TaskType.EXPLAIN: [
                r"explain", r"describe", r"why", r"how", r"meaning of",
                r"説明", r"解説", r"なぜ", r"どういうこと", r"意味"
            ]
        }
        
        self._domain_indicators = {
            MathDomain.CALCULUS: [r"derive", r"integrate", r"derivative", r"integral", r"limit", r"diff", r"微分", r"積分", r"極限"],
            MathDomain.ALGEBRA: [r"solve for", r"simplify", r"factorize", r"factor", r"polynomial", r"equation", r"方程式", r"因数分解", r"展開"],
            MathDomain.LINEAR_ALGEBRA: [r"matrix", r"vector", r"eigen", r"determinant", r"行列", r"ベクトル", r"固有値"],
            MathDomain.GEOMETRY: [r"area", r"volume", r"angle", r"triangle", r"circle", r"面積", r"体積", r"角度", r"三角形", r"円"]
        }

    def parse(self, text: str) -> SemanticIR:
        """
        Parses natural language text into Semantic Intermediate Representation.
        """
        original_text = text.strip()
        
        # 1. Detect Intent (Task)
        task, confidence = self._detect_task(original_text)
        
        # 2. Extract Mathematical Expression (Heuristic)
        # We assume the user might mix text and math. 
        # Strategy: Look for the longest contiguous string that parses effectively, 
        # or use regex to strip common words and leave the rest.
        # For Phase 1, we aggressively attempt to parse the whole string as valid input if possible,
        # OR separate by looking for non-text characters.
        
        cleaned_text, extracted_math = self._extract_math_candidates(original_text)
        
        # 3. Detect Domain
        domain = self._detect_domain(cleaned_text + " " + " ".join(extracted_math))
        
        # 4. Infer Goal based on task and keywords
        goal = self._infer_goal(task, cleaned_text)
        
        # 5. Construct Inputs
        inputs = []
        for math_str in extracted_math:
            # Normalize specific symbols
            normalized_expr = CoherentInputParser.normalize(math_str)
            if normalized_expr:
                inputs.append(InputItem(type="expression", value=normalized_expr))

        # 6. Metadata
        is_japanese = any(ord(c) > 128 for c in original_text) # Simple heuristic
        lang = "ja" if is_japanese else "en"
        
        meta = LanguageMeta(
            original_language=lang, 
            original_text=original_text,
            detected_intent_confidence=confidence
        )
        
        return SemanticIR(
            task=task,
            math_domain=domain,
            goal=goal,
            inputs=inputs,
            constraints=Constraints(),
            language_meta=meta
        )

    def _detect_task(self, text: str) -> Tuple[TaskType, float]:
        text_lower = text.lower()
        
        # 1. Check Keywords first (Priority)
        for task, patterns in self._intent_patterns.items():
            for pattern in patterns:
                # Use word boundaries or strict matching to avoid partial words if needed
                if re.search(pattern, text_lower):
                    return task, 1.0 # High confidence on keyword match

        # 2. Heuristic: If it looks like pure math (mostly numbers/symbols), assume SOLVE
        # We exclude common non-math words if possible, but matching \w is risky.
        # Let's check if it contains ANY intent keywords first (handled above).
        # If no intent keywords, and it has math symbols, assume SOLVE.
        
        # Check for presence of distinct math operators or equals to boost confidence
        if re.search(r'[=+\-*/^]', text):
             return TaskType.SOLVE, 0.8

        # Fallback default
        return TaskType.SOLVE, 0.5 

    def _detect_domain(self, text: str) -> MathDomain:
        text_lower = text.lower()
        for domain, patterns in self._domain_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return domain
        return MathDomain.ARITHMETIC # Default fallback

    def _infer_goal(self, task: TaskType, text: str) -> Optional[GoalType]:
        text_lower = text.lower()
        if task == TaskType.VERIFY:
            return None
            
        if "graph" in text_lower or "plot" in text_lower or "グラフ" in text_lower:
            return GoalType.GRAPH
            
        if "prove" in text_lower or "proof" in text_lower or "証明" in text_lower:
            return GoalType.PROOF
            
        if "simplify" in text_lower or "factor" in text_lower or "expand" in text_lower or "簡単" in text_lower or "因数分解" in text_lower:
            return GoalType.TRANSFORMATION
            
        return GoalType.FINAL_VALUE

    def _extract_math_candidates(self, text: str) -> Tuple[str, List[str]]:
        """
        Separates natural language instructions from mathematical expressions.
        """
        candidates = []
        
        # Refined Heuristic 1: Equation Extraction
        # Look for typical equation structure: LHS = RHS
        # Allow spaces around =. 
        # Don't matching "Solve " as part of it.
        # We can look for substrings that contain `=` and check their edges.
        
        if "=" in text:
            # Simple split by known keywords to isolate the equation part?
            # Or use a regex that requires operators/vars.
            
            # This regex attempts to find specific equation-like patterns:
            # (something like x^2...) = (something)
            # We explicitly exclude standard English words in the lookaround if possible, but regex is weak there.
            
            # Revised Regex:
            # Capture characters commonly in math: [a-zA-Z0-9_.()+\-*/^]
            # Must include '='.
            # \b might be useful.
            
            # Let's try: Find the '='. Expand left and right until we hit non-math characters (like words).
            # But "x" is a word. 
            
            # Heuristic: Remove the command keywords, treat the rest as math?
            # "Solve x^2+1=0" -> remove "Solve" -> "x^2+1=0"
            pass # We will do this removal below universally

        # Universal fallback strategy: Remove known intent keywords and clean up.
        cleaned_for_extraction = text
        all_keywords = [p for patterns in self._intent_patterns.values() for p in patterns]
        # Sort keywords by length descending to remove longest phrases first
        all_keywords.sort(key=len, reverse=True)
        
        for kw in all_keywords:
            # Use word boundary if keyword starts/ends with alphanumeric
            pattern = kw
            if re.match(r'^[a-zA-Z0-9]', kw):
                pattern = r'\b' + pattern
            if re.search(r'[a-zA-Z0-9]$', kw):
                pattern = pattern + r'\b'
            cleaned_for_extraction = re.sub(pattern, " ", cleaned_for_extraction, flags=re.IGNORECASE)
        
        # Remove domain keywords too
        all_domain = [p for patterns in self._domain_indicators.values() for p in patterns]
        # Sort by length
        all_domain.sort(key=len, reverse=True)
        
        for kw in all_domain:
             pattern = kw
             if re.match(r'^[a-zA-Z0-9]', kw):
                pattern = r'\b' + pattern
             if re.search(r'[a-zA-Z0-9]$', kw):
                pattern = pattern + r'\b'
             cleaned_for_extraction = re.sub(pattern, " ", cleaned_for_extraction, flags=re.IGNORECASE)

        # Japanese specific cleanup: Remove all Hiragana and common punctuation
        # This handles particles like を, は, and auxiliary endings like ください left over.
        # Range \u3040-\u309F is Hiragana. \u3000-\u303F is CJK Symbols and Punctuation.
        cleaned_for_extraction = re.sub(r'[\u3040-\u309F\u3000-\u303F]+', ' ', cleaned_for_extraction)

        # What remains should be the math. 
        # " x^2 + 2*x + 1 = 0 "
        extracted = cleaned_for_extraction.strip()
        
        # Sanity check: does it look like math?
        # Must contain at least one digit or operator or known variable?
        if len(extracted) > 0:
             candidates.append(extracted)
             
        return text.replace(extracted, " "), candidates

