import numpy as np

class SubjectiveHandler:
    def __init__(self):
        self.observations = []

    def record_observation(self, material_name: str, category: str, intensity: float, description: str, conditions: dict = None) -> dict:
        obs = {
            "material_name": material_name,
            "category": category,
            "intensity": intensity,
            "description": description,
            "conditions": conditions or {}
        }
        self.observations.append(obs)
        return obs

    def compute_subjective_score(self, obs_a: list[dict], obs_b: list[dict]) -> float:
        return 0.5

    def encode_observations(self, observations: list[dict]) -> np.ndarray:
        return np.zeros(10)

if __name__ == "__main__":
    handler = SubjectiveHandler()
    handler.record_observation("Quartz", "tactile", 0.8, "Cold and smooth")
    print("Recorded subjective observation.")
