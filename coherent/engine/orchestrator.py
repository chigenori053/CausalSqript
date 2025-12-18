from typing import Dict, Any, Type
import importlib

from coherent.engine.interfaces import BaseModule
from coherent.engine.context_manager import ContextManager
from coherent.engine.parser import Parser
from coherent.engine.errors import EvaluationError, SyntaxError as DSLSyntaxError

# Interface for Modules (Parser + Engine)
# Since modules are packages/directories, we expect them to export 'parser' and 'engine' classes 
# or instances. We adapt here.

class CoreOrchestrator:
    """
    The Command Tower of the Coherent Core.
    Routes execution to the appropriate Domain Module based on the current mode.
    """
    
    def __init__(self):
        self.context_manager = ContextManager()
        self.modules: Dict[str, Any] = {} # Key: mode name, Value: Module instance
        self.current_mode = "Algebra" # Default mode
        
        # Registry mapping Mode Name -> Module Package Path
        self._module_registry = {
            "Arithmetic": "coherent.engine.modules.arithmetic",
            "Calculus": "coherent.engine.modules.calculus",
            "Geometry": "coherent.engine.modules.geometry",
            # "Algebra": "coherent.engine.modules.algebra", # To be implemented
        }

    def set_mode(self, mode: str):
        """Switch the computation mode, loading the module if necessary."""
        if mode not in self.modules:
            self._load_module(mode)
            
        self.current_mode = mode
        self.context_manager.switch_context(mode)

    def execute_script(self, script_text: str) -> Dict[str, Any]:
        """
        Parse and execute a full Coherent DSL script.
        This handles Problem definitions which might set the mode.
        """
        parser = Parser(script_text)
        program = parser.parse()
        
        results = {}
        
        # 1. Handle Problem Node(s)
        for node in program.body:
            if hasattr(node, 'mode') and node.mode:
                self.set_mode(node.mode)
            
            # If it's a ProblemNode, we might evaluate it? 
            # Typically Problem defines the expression to be solved.
            # StepNodes define intermediate steps.
            
            # For Phase 2, we simulate Step execution if script implies it.
            # But normally execute_step is called interactively or via script runner.
            pass
            
        return results

    def execute_step(self, expr_str: str) -> Any:
        """
        Execute a single step expression using the current mode's module.
        """
        if self.current_mode not in self.modules:
             # Fallback or error if default mode not loaded?
             # For Phase 2, if Algebra is default but not impl, we fail unless switched to Arithmetic
             raise EvaluationError(f"Module for mode '{self.current_mode}' is not loaded or implemented.")
             
        module = self.modules[self.current_mode]
        
        # 1. Parse
        ast = module['parser'].parse(expr_str)
        
        # 2. Evaluate
        result = module['engine'].evaluate(ast, self.context_manager.current_context)
        
        return result

    def _load_module(self, mode: str):
        """Dynamically load a domain module."""
        package_path = self._module_registry.get(mode)
        if not package_path:
             raise ValueError(f"Unknown mode: {mode}")
             
        try:
            mod = importlib.import_module(package_path)
            
            # Expecting the module package to expose Parser and Engine classes
            # conventionally in parser.py and engine.py
            
            parser_mod = importlib.import_module(f"{package_path}.parser")
            engine_mod = importlib.import_module(f"{package_path}.engine")
            
            # Instantiate. Usually expected class names: ArithmeticParser, FastMathEngine
            # But interfaces might vary. Let's assume arbitrary naming or convention.
            # Convention: {Mode}Parser, {Mode}Engine ?? Or just look for subclasses of Base?
            
            # Simple convention for Phase 2:
            # arithmetic.parser.ArithmeticParser
            # arithmetic.engine.FastMathEngine
            
            parser_cls_name = f"{mode}Parser"
            engine_cls_name = "FastMathEngine" if mode == "Arithmetic" else f"{mode}Engine"
            
            parser_cls = getattr(parser_mod, parser_cls_name)
            engine_cls = getattr(engine_mod, engine_cls_name)
            
            self.modules[mode] = {
                'parser': parser_cls(),
                'engine': engine_cls()
            }
            
        except (ImportError, AttributeError) as e:
            raise EvaluationError(f"Failed to load module for mode '{mode}': {e}")
