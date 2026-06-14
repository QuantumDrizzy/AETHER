import numpy as np

class MetricCalculator:
    @staticmethod
    def lattice_mismatch(mat_a: dict, mat_b: dict) -> float:
        try:
            a1 = mat_a.get("crystal", {}).get("lattice_params", {}).get("a", None)
            a2 = mat_b.get("crystal", {}).get("lattice_params", {}).get("a", None)
            if a1 is None or a2 is None:
                return 0.5
            diff = abs(a1 - a2)
            avg = (a1 + a2) / 2.0
            mismatch = diff / avg
            return max(0.0, 1.0 - mismatch * 10.0) # 10% mismatch means 0 score
        except Exception:
            return 0.5

    @staticmethod
    def thermal_expansion_mismatch(mat_a: dict, mat_b: dict) -> float:
        try:
            t1 = mat_a.get("physical", {}).get("thermal_expansion", None)
            t2 = mat_b.get("physical", {}).get("thermal_expansion", None)
            if t1 is None or t2 is None:
                return 0.5
            diff = abs(t1 - t2)
            # Assuming typical CTEs are ~1e-6 to 1e-5
            return max(0.0, np.exp(-diff * 1e5))
        except Exception:
            return 0.5

    @staticmethod
    def impedance_matching(mat_a: dict, mat_b: dict) -> float:
        return 0.5

    @staticmethod
    def resonance_overlap(mat_a: dict, mat_b: dict) -> float:
        return 0.5

    @staticmethod
    def piezoelectric_coupling(mat_a: dict, mat_b: dict) -> float:
        return 0.5

    @staticmethod
    def mechanical_compatibility(mat_a: dict, mat_b: dict) -> float:
        return 0.5

    @staticmethod
    def bandgap_alignment(mat_a: dict, mat_b: dict) -> float:
        return 0.5

if __name__ == "__main__":
    mat_a = {"name": "Quartz", "crystal": {"lattice_params": {"a": 4.914}}, "physical": {"thermal_expansion": 13.2e-6}}
    mat_b = {"name": "BaTiO3", "crystal": {"lattice_params": {"a": 3.992}}, "physical": {"thermal_expansion": 14.0e-6}}
    print(f"Lattice mismatch score: {MetricCalculator.lattice_mismatch(mat_a, mat_b)}")
    print(f"Thermal mismatch score: {MetricCalculator.thermal_expansion_mismatch(mat_a, mat_b)}")
