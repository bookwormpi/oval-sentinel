"""Experimental neural movement and aim with automatic weapon and loot actions."""
import json
import math
from pathlib import Path
import zlib

from agent import MechRoyaleBot
from combat import combat_action
from geometry import clamp, wrap_angle


MODEL_PATH = Path(__file__).resolve().parent / "neural_policy.json"
ARCHITECTURE = "hashed-observation-mlp-v1"
OBS_BUCKETS = 256
CONFIG_BUCKETS = 64
# Observed entities come straight from the protocol observation; static map groups
# are indexed once from the map files (see map_index) since they never move.
ENTITY_GROUPS = ("mechs", "projectiles", "crates", "cubes", "pickups", "drops")
MAP_GROUPS = ("prisms", "trees", "cars", "barriers")
GROUPS = ENTITY_GROUPS + MAP_GROUPS
NEAREST = 4
GLOBAL_SIZE = 15
INPUT_SIZE = OBS_BUCKETS + CONFIG_BUCKETS + GLOBAL_SIZE + len(GROUPS) * (NEAREST + 1)
HIDDEN_SIZE = 16
OUTPUT_SIZE = 4
WEIGHT_COUNT = HIDDEN_SIZE * (INPUT_SIZE + 1) + OUTPUT_SIZE * (HIDDEN_SIZE + 1)


def _bucket(values, name, value):
    digest = zlib.crc32(name.encode("utf-8"))
    values[digest % len(values)] += value if digest & 0x80000000 else -value


def _number_scale(path, config):
    leaf = path.rsplit(".", 1)[-1]
    if leaf in ("id", "by", "target", "holder"):
        return 20000.0
    if leaf == "tick":
        return max(1.0, float(config["match"]["duration_s"]) * float(config["match"]["tick_hz"]))
    if leaf in ("t", "age", "expires", "shrink_start", "shrink_end", "lands_in"):
        return max(1.0, float(config["match"]["duration_s"]))
    if ".aim." in path or leaf == "bearing":
        return math.pi
    if ".pos." in path or ".center." in path or ".next_center." in path or ".path." in path:
        return max(1.0, float(config["zone"]["start_radius"]))
    if ".vel." in path:
        return max(1.0, float(config["weapons"]["gun"]["speed"]))
    if leaf in ("hp", "max_hp", "amount", "damage"):
        return max(1.0, float(config["mech"]["hp"]))
    if leaf in ("energy", "max_energy"):
        return max(1.0, float(config["mech"]["energy"]))
    if leaf in ("radius", "next_radius", "range"):
        return max(1.0, float(config["zone"]["start_radius"]))
    return 100.0


def _encode(value, path, buckets, config):
    if isinstance(value, dict):
        _bucket(buckets, path + "#object", 1.0)
        for key in sorted(value):
            _encode(value[key], f"{path}.{key}", buckets, config)
    elif isinstance(value, list):
        _bucket(buckets, path + "#array", 1.0)
        _bucket(buckets, path + "#length", math.tanh(len(value) / 10.0))
        for index, item in enumerate(value):
            _encode(item, f"{path}.{index}", buckets, config)
    elif isinstance(value, bool):
        _bucket(buckets, path, 1.0 if value else -1.0)
    elif value is None:
        _bucket(buckets, path + "=null", 1.0)
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite neural input at {path}")
        _bucket(buckets, path + "#present", 1.0)
        _bucket(buckets, path, math.tanh(value / _number_scale(path, config)))
    elif isinstance(value, str):
        _bucket(buckets, path + "=" + value, 1.0)
    else:
        raise TypeError(f"Unsupported neural input at {path}: {type(value).__name__}")


def _compressed(value, name, count, config):
    buckets = [0.0] * count
    _encode(value, name, buckets, config)
    return [math.tanh(v / 3.0) for v in buckets]


def map_index(world):
    cached = getattr(world, "_neural_map_index", None)
    if cached is not None:
        return cached
    root = getattr(world, "root", None)
    objects, grid = [], {}

    def add(group, bounds):
        index = len(objects)
        objects.append((group, bounds))
        for gx in range(math.floor(bounds[0] / 50), math.floor(bounds[2] / 50) + 1):
            for gy in range(math.floor(bounds[1] / 50), math.floor(bounds[3] / 50) + 1):
                grid.setdefault((gx, gy), []).append(index)

    if root is not None:
        # The World class already parsed prism rings into bounds/z-bounds; reuse
        # them instead of re-reading and re-parsing the (hundreds-of-KB) map file.
        # Prisms with no ring get a placeholder (0, 0, 0, 0) bounding box there; skip
        # those rather than indexing a bogus zero-sized prism at the map origin.
        for prism, (min_x, min_y, max_x, max_y), (base, top) in zip(
                getattr(world, "prisms", []), getattr(world, "prism_bounds", []),
                getattr(world, "prism_zbounds", [])):
            if prism.get("ring"):
                add("prisms", (min_x, min_y, max_x, max_y, base, top))
        with (Path(root) / "v2" / "world.json").open(encoding="utf-8") as source:
            data = json.load(source)
        for group in ("trees", "cars"):
            packed = data[group]
            fields = packed["fields"]
            values = packed["data"]
            for offset in range(0, len(values), len(fields)):
                obj = dict(zip(fields, values[offset:offset + len(fields)]))
                if group == "trees":
                    radius = max(obj["trunkR"], obj["crownR"])
                    min_x = min(obj["x"], obj["crownX"]) - radius
                    max_x = max(obj["x"], obj["crownX"]) + radius
                    min_y = min(obj["y"], obj["crownY"]) - radius
                    max_y = max(obj["y"], obj["crownY"]) + radius
                else:
                    radius = math.hypot(obj["length"], obj["width"]) / 2
                    min_x, max_x = obj["x"] - radius, obj["x"] + radius
                    min_y, max_y = obj["y"] - radius, obj["y"] + radius
                add(group, (min_x, min_y, max_x, max_y, obj["z"], obj["z"] + obj["height"]))
        for barrier in data["barriers"]:
            line = barrier["line"]
            if line:
                base = min(world.ground(line[i], line[i + 1]) for i in range(0, len(line), 2))
                add("barriers", (min(line[0::2]), min(line[1::2]), max(line[0::2]), max(line[1::2]),
                                 base, base + barrier["height"]))
    world._neural_map_index = (objects, grid)
    return world._neural_map_index


def _map_distances(index, x, y, z, radius):
    objects, grid = index
    candidates = set()
    for gx in range(math.floor((x - radius) / 50), math.floor((x + radius) / 50) + 1):
        for gy in range(math.floor((y - radius) / 50), math.floor((y + radius) / 50) + 1):
            candidates.update(grid.get((gx, gy), ()))
    result = {group: [] for group in MAP_GROUPS}
    for candidate in candidates:
        group, (min_x, min_y, max_x, max_y, min_z, max_z) = objects[candidate]
        distance = math.sqrt(max(min_x - x, 0, x - max_x) ** 2
                             + max(min_y - y, 0, y - max_y) ** 2
                             + max(min_z - z, 0, z - max_z) ** 2)
        if distance <= radius:
            result[group].append(distance)
    return result


def encode_observation(obs, config, config_features, object_radius, static_index=None):
    me = obs["self"]
    x, y, z = (float(v) for v in me["pos"])
    yaw, pitch = (float(v) for v in me["aim"])
    vx, vy, vz = (float(v) for v in me["vel"])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    scale = max(1.0, float(config["zone"]["start_radius"]))
    speed = max(1.0, float(config["mech"]["boost_speed"]))
    center = config["zone"]["start_center"]
    globals_ = [
        math.tanh((x - center[0]) / scale), math.tanh((y - center[1]) / scale), math.tanh(z / scale),
        math.tanh((vx * cos_yaw + vy * sin_yaw) / speed),
        math.tanh((-vx * sin_yaw + vy * cos_yaw) / speed), math.tanh(vz / speed),
        cos_yaw, sin_yaw, math.tanh(pitch / math.pi),
        clamp(float(me["hp"]) / max(1.0, float(me["max_hp"])), 0.0, 1.0),
        clamp(float(me["energy"]) / max(1.0, float(me["max_energy"])), 0.0, 1.0),
        math.tanh(float(me["cubes"]) / max(1.0, float(config["cubes"]["stat_cap"]))),
        clamp(float(obs["t"]) / max(1.0, float(config["match"]["duration_s"])), 0.0, 1.0),
        clamp(float(obs["alive"]) / max(1.0, float(config["match"]["mechs"])), 0.0, 1.0),
        1.0 if me["in_zone"] else -1.0,
    ]
    buckets = [0.0] * OBS_BUCKETS
    _encode(obs, "obs", buckets, config)
    distances = []
    map_distances = _map_distances(static_index, x, y, z, object_radius) if static_index else {}
    for group in GROUPS:
        nearby = map_distances.get(group, [])
        if group not in map_distances:
            for item in obs.get(group, []):
                distance = math.dist((x, y, z), item["pos"])
                if distance <= object_radius:
                    nearby.append(distance)
        nearby.sort()
        distances.extend([1.0 - d / object_radius for d in nearby[:NEAREST]])
        distances.extend([0.0] * (NEAREST - min(len(nearby), NEAREST)))
        distances.append(len(nearby) / (len(nearby) + NEAREST))
        for index, distance in enumerate(nearby):
            _bucket(buckets, f"distance.{group}.{index}", 1.0 - distance / object_radius)
    result = globals_ + distances + [math.tanh(v / 3.0) for v in buckets] + config_features
    if len(result) != INPUT_SIZE or not all(math.isfinite(v) and -1 <= v <= 1 for v in result):
        raise ValueError("Invalid normalized neural input")
    return result


def validate_model(model):
    if not isinstance(model, dict) or model.get("version") != 1 or model.get("architecture") != ARCHITECTURE:
        raise ValueError("Invalid neural policy format or architecture")
    radius = model.get("object_radius")
    if (isinstance(radius, bool) or not isinstance(radius, (int, float))
            or not math.isfinite(radius) or not 0 < radius <= 5000):
        raise ValueError("Neural object radius must be finite and in (0, 5000]")
    weights = model.get("weights")
    if not isinstance(weights, list) or len(weights) != WEIGHT_COUNT:
        raise ValueError(f"Neural policy requires exactly {WEIGHT_COUNT} weights")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or abs(v) > 50 for v in weights):
        raise ValueError("Neural weights must be finite numbers in [-50, 50]")
    return model


def load_model(path=MODEL_PATH):
    with open(path, encoding="utf-8") as source:
        return validate_model(json.load(source))


def random_model(rng, object_radius):
    weights = []
    for _ in range(HIDDEN_SIZE):
        weights.extend(rng.gauss(0, 1 / math.sqrt(INPUT_SIZE)) for _ in range(INPUT_SIZE))
        weights.append(0.0)
    for _ in range(OUTPUT_SIZE):
        weights.extend(rng.gauss(0, 0.3 / math.sqrt(HIDDEN_SIZE)) for _ in range(HIDDEN_SIZE))
        weights.append(0.0)
    return validate_model({"version": 1, "architecture": ARCHITECTURE,
                           "object_radius": object_radius, "weights": weights})


def forward(inputs, weights):
    hidden = []
    offset = 0
    for _ in range(HIDDEN_SIZE):
        total = weights[offset + INPUT_SIZE]
        total += sum(inputs[i] * weights[offset + i] for i in range(INPUT_SIZE))
        hidden.append(math.tanh(total))
        offset += INPUT_SIZE + 1
    output = []
    for _ in range(OUTPUT_SIZE):
        total = weights[offset + HIDDEN_SIZE]
        total += sum(hidden[i] * weights[offset + i] for i in range(HIDDEN_SIZE))
        output.append(math.tanh(total))
        offset += HIDDEN_SIZE + 1
    return output


class NeuralBot:
    name = "Oval Sentinel neural"

    def __init__(self, hello, world=None, model=None):
        self.config = hello["config"]
        self.model = validate_model(model) if model is not None else load_model()
        hello_features = {key: value for key, value in hello.items() if key != "map"}
        self.config_features = _compressed(hello_features, "hello", CONFIG_BUCKETS, self.config)
        self.automatic = MechRoyaleBot(hello, strategy="vanguard", world=world)
        self.static_index = map_index(self.automatic.world)

    def act(self, obs):
        action = self.automatic.act(obs)
        inputs = encode_observation(obs, self.config, self.config_features, self.model["object_radius"],
                                    self.static_index)
        forward_move, left_move, yaw_offset, pitch_offset = forward(inputs, self.model["weights"])
        yaw, pitch = obs["self"]["aim"]
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        mx = forward_move * cos_yaw - left_move * sin_yaw
        my = forward_move * sin_yaw + left_move * cos_yaw
        length = math.hypot(mx, my)
        if length > 1:
            mx, my = mx / length, my / length
        stats = obs["self"].get("stats", self.config["mech"])
        minimum, maximum = float(stats["min_pitch"]), float(stats["max_pitch"])
        commanded = [wrap_angle(yaw + yaw_offset * math.pi),
                     clamp(pitch + pitch_offset * (maximum - minimum) / 2, minimum, maximum)]
        target = self.automatic.current_combat_target
        _, fire, _, _ = combat_action(obs, obs["self"], target, self.config, self.automatic.world,
                                      correct_mortar=True, precision=True, commanded_aim=commanded)
        if target is None:
            for slot, weapon in enumerate(obs["self"]["weapons"][:2]):
                if weapon and weapon["kind"] == "laser" and float(weapon.get("charge", 0)) > 0:
                    fire[slot] = True
        action["fire"] = fire
        action["move"] = [round(mx, 5), round(my, 5)]
        action["aim"] = [round(commanded[0], 5), round(commanded[1], 5)]
        return action
