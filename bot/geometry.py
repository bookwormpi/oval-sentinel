"""Small, allocation-conscious 2D/3D geometry helpers."""
from __future__ import annotations
import math

TAU = 2.0 * math.pi

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def wrap_angle(a):
    return (a + math.pi) % TAU - math.pi

def angle_delta(a, b):
    return wrap_angle(a - b)

def unit2(x, y):
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-9 else (0.0, 0.0)

def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def dist3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def point_segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    den = dx*dx + dy*dy
    if den <= 1e-12:
        return dist2(p, a)
    t = clamp(((p[0]-a[0])*dx + (p[1]-a[1])*dy) / den, 0.0, 1.0)
    return math.hypot(p[0]-(a[0]+t*dx), p[1]-(a[1]+t*dy))

def point_in_ring(x, y, ring):
    inside = False
    n = len(ring) // 2
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[2*i], ring[2*i+1]
        xj, yj = ring[2*j], ring[2*j+1]
        if ((yi > y) != (yj > y)):
            cross = (xj-xi) * (y-yi) / ((yj-yi) or 1e-15) + xi
            if x < cross:
                inside = not inside
        j = i
    return inside

def point_in_prism(x, y, z, prism, margin=0.0):
    if z < prism.get("base", -1e9)-margin or z > prism.get("top", 1e9)+margin:
        return False
    ring = prism.get("ring", [])
    if not point_in_ring(x, y, ring):
        return False
    for hole in prism.get("holes", []):
        if point_in_ring(x, y, hole):
            return False
    return True

def segment_hits_prism(a, b, prism, radius=2.5):
    """Conservative sampled cylinder-vs-building test for a horizontal route."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    length = math.hypot(dx, dy)
    steps = max(2, int(length / 2.5) + 1)
    z = min(a[2], b[2]) + 4.5
    # Mech center must not intersect a prism at the sampled altitude.
    for i in range(steps + 1):
        t = i / steps
        x, y = a[0] + dx*t, a[1] + dy*t
        if point_in_prism(x, y, z, prism):
            return True
        # Check footprint proximity to edges, with the mech radius.
        ring = prism.get("ring", [])
        if z < prism.get("base", -1e9) or z > prism.get("top", 1e9):
            continue
        if point_in_ring(x, y, ring):
            in_hole = any(point_in_ring(x, y, h) for h in prism.get("holes", []))
            if not in_hole:
                return True
        n = len(ring)//2
        for j in range(n):
            p = (ring[2*j], ring[2*j+1])
            q = (ring[2*((j+1)%n)], ring[2*((j+1)%n)+1])
            if point_segment_distance((x,y), p, q) < radius:
                # Ignore an edge if this sample is in a footprint hole.
                if not any(point_in_ring(x, y, h) for h in prism.get("holes", [])):
                    return True
    return False

def intercept_time(r, v, speed):
    """Earliest positive projectile intercept time; zero means no valid lead."""
    a = sum(q*q for q in v) - speed*speed
    b = 2.0 * sum(r[i]*v[i] for i in range(3))
    c = sum(q*q for q in r)
    if abs(a) < 1e-9:
        return max(0.0, -c/b) if b < -1e-9 else 0.0
    d = b*b - 4*a*c
    if d < 0:
        return 0.0
    root = math.sqrt(d)
    ts = [t for t in ((-b-root)/(2*a), (-b+root)/(2*a)) if t > 0]
    return min(ts) if ts else 0.0

def ballistic_pitch(horizontal, dz, speed, gravity, high_arc=False):
    """Solve projectile launch pitch. Return None if out of ballistic range."""
    if horizontal < 1e-6 or speed <= 0 or gravity <= 0:
        return math.pi/2 if dz > 0 else -math.pi/2
    v2 = speed*speed
    disc = v2*v2 - gravity*(gravity*horizontal*horizontal + 2*dz*v2)
    if disc < 0:
        return None
    root = math.sqrt(disc)
    tan = (v2 + (root if high_arc else -root)) / (gravity*horizontal)
    return math.atan(tan)
