# Implementation Directive: Category-Aware Optimization (Knowledge & Rendering)
**Status**: Draft
**Target**: Coherent Core Architecture

## Goal
Leverage `MathCategory` to optimize `KnowledgeRegistry` search efficiency and enhance `LogFormatter` expressiveness.

## 1. Architecture Overview

| Component | Change | Purpose |
| :--- | :--- | :--- |
| `core/knowledge_registry.py` | Add `category_filter` to `match` method | Exclude irrelevant rules to improve search speed and precision. |
| `core/core_runtime.py` | Add `category` metadata to logs | Record the category of calculation steps for the renderer. |
| `core/renderers.py` | **[New]** Implement category-specific formatters | Generate specialized string representations based on category (e.g., Geometry, Calculus). |
| `core/log_formatter.py` | Use `ContentRenderer` | Delegate message formatting to the new renderer. |

## 2. Implementation Steps

### Step 1: KnowledgeRegistry Optimization (Rule Filtering)
Add category-based filtering to `KnowledgeRegistry`.

**File**: `core/knowledge_registry.py`

```python
# [Imports]
from typing import Optional, List, Set
# ...

class KnowledgeRegistry:
    # ... (existing init)

    def match(
        self, 
        before: str, 
        after: str, 
        category: str | None = None  # Added: Category hint
    ) -> Optional[KnowledgeNode]:
        """
        Identifies the rule that transforms 'before' into 'after'.
        Optimized by category filtering.
        """
        # Determine allowed domains based on category
        allowed_domains = self._resolve_domains(category)
        
        # 1. Search through Maps (Priority Search)
        for rule_map in self.maps:
            for rule_id in rule_map.rules:
                node = self.rules_by_id.get(rule_id)
                if not node:
                    continue
                
                # --- Filtering ---
                if allowed_domains and node.domain not in allowed_domains:
                    continue
                # -----------------

                match = self._match_node(node, before, after, is_before_numeric=False)
                if match:
                    return match

        # 2. Fallback (Unmapped rules)
        # ... (Apply filtering similarly)
        return None

    def _resolve_domains(self, category: str | None) -> Set[str] | None:
        """Returns the set of domains to search based on MathCategory."""
        if not category:
            return None
            
        # Always allow basic arithmetic and algebra
        commons = {"universal", "arithmetic", "algebra"}
        
        if category == "geometry":
            return commons | {"geometry"}
        elif category == "calculus":
            return commons | {"calculus", "analysis"}
        elif category == "linear_algebra":
            return commons | {"linear_algebra", "matrix"}
        elif category == "statistics":
            return commons | {"statistics", "probability"}
        
        return commons
```

### Step 2: CoreRuntime Integration (Context Injection)
Pass the detected category to `KnowledgeRegistry` and `LearningLogger` in `check_step`.

**File**: `core/core_runtime.py`

```python
    def check_step(self, expr: str) -> dict:
        # ... 
        
        # Pass category to KnowledgeRegistry
        rule_id: str | None = None
        rule_meta: dict[str, Any] | None = None
        
        if is_valid and self.knowledge_registry is not None:
            # Use detected category
            current_cat = self._current_category.value if self._current_category else None
            
            matched = self.knowledge_registry.match(
                before, 
                after, 
                category=current_cat  # Pass category
            )
            if matched:
                rule_id = matched.id
                rule_meta = matched.to_metadata()

        # ...

        # Include category in result metadata (for rendering)
        result["details"]["category"] = self._current_category.value
        
        return result
```

### Step 3: Category-Specific Rendering (Log Rendering)
Implement a renderer class to format log messages based on category.

**File**: `core/renderers.py` (New)

```python
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class RenderContext:
    expression: str
    category: str
    metadata: Dict[str, Any]

class ContentRenderer:
    """Formats log messages based on category."""

    @staticmethod
    def render_step(context: RenderContext) -> str:
        category = context.category
        expr = context.expression
        
        if category == "geometry":
            return ContentRenderer._render_geometry(expr, context.metadata)
        elif category == "calculus":
            return ContentRenderer._render_calculus(expr)
        elif category == "statistics":
            return ContentRenderer._render_statistics(expr)
        
        # Default (Algebra/Arithmetic)
        return expr

    @staticmethod
    def _render_geometry(expr: str, meta: Dict) -> str:
        if "description" in meta:
            return f"{meta['description']} ({expr})"
        return expr

    @staticmethod
    def _render_calculus(expr: str) -> str:
        # Example: 'diff(x**2, x)' -> "d/dx (x^2)"
        if "diff(" in expr:
            return expr.replace("diff(", "d/dx(").replace(",", ", ")
        return expr

    @staticmethod
    def _render_statistics(expr: str) -> str:
        return f"📊 {expr}"
```

**File**: `core/log_formatter.py` (Modify)

Call `ContentRenderer` from the existing formatter.

```python
from .renderers import ContentRenderer, RenderContext

def format_record_message(record: dict, *, include_meta: bool = True) -> List[str]:
    # ...
    
    # Get category from metadata
    meta = record.get("meta") or {}
    category = meta.get("category", "algebra")
    
    # Render
    ctx = RenderContext(
        expression=_expression_for(record),
        category=category,
        metadata=meta
    )
    rendered_expr = ContentRenderer.render_step(ctx)
    
    base = f"{label}: {rendered_expr}".rstrip()
    
    # ...
```

## 3. Benefits
- **Performance**: Exclude irrelevant rules (e.g., complex algebra when solving geometry).
- **Conflict Avoidance**: Prioritize appropriate rules for the category.
- **Educational Value**: Display category-specific notation (e.g., Calculus, Statistics).

## 4. Action Items
- [ ] Update `core/knowledge_registry.py`: `match` signature and filtering.
- [ ] Update `core/core_runtime.py`: Pass category in `check_step`.
- [ ] Create `core/renderers.py`: Implement rendering logic.
- [ ] Update `core/log_formatter.py`: Integrate `ContentRenderer`.
- [ ] Create `tests/test_category_rendering.py`: Verify rendering.
