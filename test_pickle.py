import pickle
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.geometry_engine import GeometryEngine
from coherent.engine.computation_engine import ComputationEngine

def test_pickle():
    sym_engine = SymbolicEngine()
    try:
        pickle.dumps(sym_engine)
        print("SymbolicEngine is picklable")
    except Exception as e:
        print(f"SymbolicEngine is NOT picklable: {e}")

    try:
        geo_engine = GeometryEngine()
        pickle.dumps(geo_engine)
        print("GeometryEngine is picklable")
    except Exception as e:
        print(f"GeometryEngine is NOT picklable: {e}")

    comp_engine = ComputationEngine(sym_engine)
    try:
        pickle.dumps(comp_engine)
        print("ComputationEngine is picklable")
    except Exception as e:
        print(f"ComputationEngine is NOT picklable: {e}")

if __name__ == "__main__":
    test_pickle()
