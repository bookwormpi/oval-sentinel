import sys, unittest
sys.path.insert(0, "bot")
from geometry import intercept_time, ballistic_pitch, point_in_ring, wrap_angle

class GeometryTests(unittest.TestCase):
    def test_intercept_stationary(self):
        self.assertAlmostEqual(intercept_time((100,0,0),(0,0,0),100), 1.0)
    def test_intercept_crossing(self):
        self.assertGreater(intercept_time((100,0,0),(0,10,0),100), 0)
    def test_ballistic_solution(self):
        p = ballistic_pitch(100, 0, 72, 22)
        self.assertIsNotNone(p)
        self.assertGreater(p, 0)
    def test_ring(self):
        ring = [0,0, 10,0, 10,10, 0,10]
        self.assertTrue(point_in_ring(5,5,ring))
        self.assertFalse(point_in_ring(15,5,ring))
    def test_angle_wrap(self):
        self.assertLess(abs(wrap_angle(3.2)), 3.2)

if __name__ == "__main__":
    unittest.main()
