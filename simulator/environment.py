"""Approximate terrain and tree visibility on top of the bot's building geometry."""
import json
import math
from pathlib import Path

from geometry import point_in_prism


def _ellipse_interval(start, end, center, radii):
    a = b = c = 0.0
    for i, radius in enumerate(radii):
        x = (start[i] - center[i]) / radius
        d = (end[i] - start[i]) / radius
        a += d * d
        b += 2 * x * d
        c += x * x
    c -= 1
    if a < 1e-12:
        return (0.0, 1.0) if c <= 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    lo, hi = max(0.0, (-b - root) / (2 * a)), min(1.0, (-b + root) / (2 * a))
    return (lo, hi) if lo <= hi else None


class Environment:
    def __init__(self, sim):
        self.sim = sim
        self.trees = []
        self.grid = {}
        self.density = 0.35
        self.block_t = 0.25
        root = getattr(sim.world, "root", None)
        if root is None:
            return
        data = json.loads((Path(root) / "v2" / "world.json").read_text())
        self.density = float(data["rules"]["canopyDensity"])
        self.block_t = float(data["rules"]["canopyBlockT"])
        trees = data["trees"]
        fields = trees["fields"]
        values = trees["data"]
        for offset in range(0, len(values), len(fields)):
            self.add_tree(dict(zip(fields, values[offset:offset + len(fields)])))

    def add_tree(self, tree):
        index = len(self.trees)
        self.trees.append(tree)
        r = max(tree["crownR"], tree["trunkR"])
        min_x = min(tree["x"], tree["crownX"]) - r
        max_x = max(tree["x"], tree["crownX"]) + r
        min_y = min(tree["y"], tree["crownY"]) - r
        max_y = max(tree["y"], tree["crownY"]) + r
        for x in range(math.floor(min_x / 50), math.floor(max_x / 50) + 1):
            for y in range(math.floor(min_y / 50), math.floor(max_y / 50) + 1):
                self.grid.setdefault((x, y), []).append(index)

    def clear(self, start, end, sight=False, canopy=True):
        world = self.sim.world
        if not world.line_of_sight(start, end):
            return False
        for prism, bounds in zip(getattr(world, "prisms", []), getattr(world, "prism_bounds", [])):
            if (bounds[0] <= end[0] <= bounds[2] and bounds[1] <= end[1] <= bounds[3]
                    and prism["base"] + 1e-6 < end[2] < prism["top"] - 1e-6
                    and point_in_prism(end[0], end[1], end[2], prism)):
                return False
        distance = math.dist(start, end)
        steps = max(1, math.ceil(distance / 3))
        for i in range(1, steps + 1):
            t = i / steps
            x, y, z = (start[j] + (end[j] - start[j]) * t for j in range(3))
            if z < world.ground(x, y) - 1e-6:
                return False
        if not sight:
            return True

        indices = set()
        for x in range(math.floor(min(start[0], end[0]) / 50), math.floor(max(start[0], end[0]) / 50) + 1):
            for y in range(math.floor(min(start[1], end[1]) / 50), math.floor(max(start[1], end[1]) / 50) + 1):
                indices.update(self.grid.get((x, y), ()))
        optical_depth = 0.0
        for index in indices:
            tree = self.trees[index]
            trunk = (_ellipse_interval(start[:2], end[:2], [tree["x"], tree["y"]],
                                       [tree["trunkR"], tree["trunkR"]])
                     if tree["trunkR"] > 0 else None)
            if trunk:
                z0 = start[2] + (end[2] - start[2]) * trunk[0]
                z1 = start[2] + (end[2] - start[2]) * trunk[1]
                if max(z0, z1) >= tree["z"] and min(z0, z1) <= tree["z"] + tree["height"]:
                    return False
            if not canopy or not self.sim.config["vision"]["canopy_blocks"]:
                continue
            vertical_radius = (tree["height"] - tree["crownBase"]) / 2
            if tree["crownR"] <= 0 or vertical_radius <= 0:
                continue
            center = [tree["crownX"], tree["crownY"], tree["z"] + tree["crownBase"] + vertical_radius]
            crown = _ellipse_interval(start, end, center, [tree["crownR"], tree["crownR"], vertical_radius])
            if crown:
                optical_depth += self.density * distance * (crown[1] - crown[0])
                if optical_depth >= -math.log(self.block_t):
                    return False
        return True

    def impact(self, start, end):
        if self.clear(start, end):
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            point = [start[i] + (end[i] - start[i]) * mid for i in range(3)]
            if self.clear(start, point):
                lo = mid
            else:
                hi = mid
        return lo
