try:
    import sympy
    print(f"SymPy version: {sympy.__version__}")
except ImportError:
    print("SymPy not found")
