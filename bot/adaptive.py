"""Validated parameters for the experimental Vanguard-derived policy."""
import json
import math
from pathlib import Path


PARAMETERS = {
    "cube_seek_distance": (100.0, 45.0, 150.0),
    "weapon_seek_distance": (85.0, 40.0, 130.0),
    "loot_threat_distance": (45.0, 20.0, 80.0),
    "crate_attack_distance": (70.0, 40.0, 120.0),
    "crate_threat_distance": (120.0, 80.0, 160.0),
    "zone_lead_seconds": (12.0, 5.0, 25.0),
}
DEFAULT_PARAMETERS = {name: values[0] for name, values in PARAMETERS.items()}
DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "adaptive_policy.json"


def validate_parameters(parameters):
    if not isinstance(parameters, dict) or set(parameters) != set(PARAMETERS):
        raise ValueError(f"Adaptive policy must define exactly: {', '.join(PARAMETERS)}")
    result = {}
    for name, (_, minimum, maximum) in PARAMETERS.items():
        value = parameters[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not minimum <= value <= maximum):
            raise ValueError(f"Adaptive policy {name} must be finite and between {minimum} and {maximum}")
        result[name] = float(value)
    return result


def load_policy(path=DEFAULT_POLICY_PATH):
    with open(path, encoding="utf-8") as source:
        policy = json.load(source)
    if not isinstance(policy, dict) or policy.get("version") != 1 or policy.get("base_strategy") != "vanguard":
        raise ValueError(f"Invalid adaptive policy format: {path}")
    return validate_parameters(policy.get("parameters"))
