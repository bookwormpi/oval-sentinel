"""Static campus map access: prism collision/LOS and terrain height sampling."""
from __future__ import annotations
import array, json, math, os
from pathlib import Path
from geometry import point_in_prism, segment_hits_prism, point_segment_distance

class World:
    def __init__(self, map_dir=None):
        self.ready = False
        self.prisms = []
        self.prism_zbounds = []
        self.los_cache = {}
        self.terrain = None
        self.nx = self.ny = 0
        self.min_x = self.min_y = self.step = 1.0
        candidates = []
        if map_dir:
            candidates.append(Path(map_dir))
        candidates += [Path("/arena"), Path(__file__).resolve().parents[1] / "arena"]
        for root in candidates:
            try:
                world_path = root / "v2" / "world.json"
                terrain_path = root / "v2" / "terrain.i16"
                if not world_path.exists():
                    continue
                data = json.loads(world_path.read_text())
                self.prisms = data.get("prisms", [])
                self.prism_bounds = []
                self.prism_zbounds = []
                for prism in self.prisms:
                    ring = prism.get("ring", [])
                    if ring:
                        xs, ys = ring[0::2], ring[1::2]
                        self.prism_bounds.append((min(xs), min(ys), max(xs), max(ys)))
                    else:
                        self.prism_bounds.append((0,0,0,0))
                    self.prism_zbounds.append((float(prism.get("base", -1e9)),
                                               float(prism.get("top", 1e9))))
                t = data["terrain"]
                self.nx, self.ny = int(t["nx"]), int(t["ny"])
                self.min_x, self.min_y, self.step = float(t["minX"]), float(t["minY"]), float(t["step"])
                if terrain_path.exists():
                    raw = terrain_path.read_bytes()
                    vals = array.array("h")
                    vals.frombytes(raw)
                    if os.sys.byteorder != "little":
                        vals.byteswap()
                    self.terrain = vals
                self.ready = True
                return
            except Exception:
                continue

    def ground(self, x, y):
        if self.terrain is None:
            return 0.0
        fx = (x-self.min_x)/self.step
        fy = (y-self.min_y)/self.step
        i, j = int(fx), int(fy)
        if i < 0 or j < 0 or i+1 >= self.nx or j+1 >= self.ny:
            return 0.0
        u, v = fx-i, fy-j
        a = self.terrain[j*self.nx+i] * 0.01
        b = self.terrain[j*self.nx+i+1] * 0.01
        c = self.terrain[(j+1)*self.nx+i] * 0.01
        d = self.terrain[(j+1)*self.nx+i+1] * 0.01
        return (a*(1-u)+b*u)*(1-v) + (c*(1-u)+d*u)*v

    def blocked(self, a, b, radius=2.5):
        if not self.ready:
            return False
        # Use a local look-ahead: long-range global paths are not needed every 100 ms.
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if length > 45.0:
            scale = 45.0 / length
            b = (a[0]+dx*scale, a[1]+dy*scale, a[2]+(b[2]-a[2])*scale)
        minx, maxx = min(a[0],b[0])-radius, max(a[0],b[0])+radius
        miny, maxy = min(a[1],b[1])-radius, max(a[1],b[1])+radius
        for p, bounds in zip(self.prisms, self.prism_bounds):
            if bounds[2] < minx or bounds[0] > maxx or bounds[3] < miny or bounds[1] > maxy:
                continue
            if segment_hits_prism(a, b, p, radius):
                return True
        return False

    def line_of_sight(self, a, b):
        if not self.ready:
            return True
        key = (
            round(a[0]), round(a[1]), round(a[2]),
            round(b[0]), round(b[1]), round(b[2]),
        )
        cached = self.los_cache.get(key)
        if cached is not None:
            return cached
        dx, dy, dz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
        distance = math.sqrt(dx*dx+dy*dy+dz*dz)
        # A 3 m sample interval is already below the mech radius. Cap long
        # sightlines to avoid work proportional to arena distance.
        n = max(3, min(96, int(distance/3.0)))
        minx, maxx = min(a[0], b[0]), max(a[0], b[0])
        miny, maxy = min(a[1], b[1]), max(a[1], b[1])
        minz, maxz = min(a[2], b[2]), max(a[2], b[2])
        for i in range(1, n):
            t = i/n
            x, y, z = a[0]+dx*t, a[1]+dy*t, a[2]+dz*t
            for j, p in enumerate(self.prisms):
                bounds = self.prism_bounds[j]
                if x < bounds[0] or x > bounds[2] or y < bounds[1] or y > bounds[3]:
                    continue
                zb = self.prism_zbounds[j]
                if z < zb[0] or z > zb[1]:
                    continue
                if point_in_prism(x, y, z, p):
                    if len(self.los_cache) >= 128:
                        self.los_cache.clear()
                    self.los_cache[key] = False
                    return False
        if len(self.los_cache) >= 128:
            self.los_cache.clear()
        self.los_cache[key] = True
        return True

    def route(self, start, goal):
        """Return a local waypoint around the first blocking prism, with bounded work."""
        if not self.blocked(start, goal):
            return goal
        dx0, dy0 = goal[0]-start[0], goal[1]-start[1]
        l0 = math.hypot(dx0, dy0)
        if l0 < 1e-6:
            return goal
        if l0 > 45.0:
            k = 45.0/l0
            goal = (start[0]+dx0*k, start[1]+dy0*k, start[2]+(goal[2]-start[2])*k)
        minx, maxx = min(start[0],goal[0])-3, max(start[0],goal[0])+3
        miny, maxy = min(start[1],goal[1])-3, max(start[1],goal[1])+3
        first = None
        for p, bounds in zip(self.prisms, self.prism_bounds):
            if bounds[2] < minx or bounds[0] > maxx or bounds[3] < miny or bounds[1] > maxy:
                continue
            if segment_hits_prism(start, goal, p):
                first = p
                break
        if first is None:
            return goal

        ring = first.get("ring", [])
        n = len(ring)//2
        if n < 3:
            return goal
        cx = sum(ring[0::2])/n
        cy = sum(ring[1::2])/n
        # Only evaluate vertices near the direct route. The old all-vertices search
        # repeatedly collision-tested every candidate against the entire campus map.
        ranked = []
        for i in range(n):
            vx, vy = ring[2*i], ring[2*i+1]
            d = point_segment_distance((vx,vy), start, goal)
            ranked.append((d, vx, vy))
        ranked.sort(key=lambda q:q[0])
        candidates = []
        for _, vx, vy in ranked[:6]:
            ox, oy = vx-cx, vy-cy
            norm = math.hypot(ox,oy) or 1.0
            for margin in (5.0, 10.0):
                qx, qy = vx+ox/norm*margin, vy+oy/norm*margin
                q = (qx, qy, self.ground(qx,qy))
                if not self.blocked(start,q) and not self.blocked(q,goal):
                    score = math.dist(start[:2],q[:2]) + math.dist(q[:2],goal[:2])
                    candidates.append((score,q))
        return min(candidates,key=lambda x:x[0])[1] if candidates else goal
