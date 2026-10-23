"""Unit tests for PPE Webcam POC core logic and utilities.

Tests pure functions:
- normalize_label()
- classify_label()
- box_area()
- region_for_person()
- overlap_fraction()
- center_in_region()
- associate_ppe()
- frame_item_status()
- smooth_status()
- overall_status()
- point_side()
- EventLogger and Dashboard stats
"""
from collections import deque
from pathlib import Path
import tempfile
import unittest

from ppe_webcam_poc import (
    normalize_label,
    classify_label,
    box_area,
    region_for_person,
    overlap_fraction,
    center_in_region,
    associate_ppe,
    frame_item_status,
    smooth_status,
    overall_status,
    point_side,
    check_zone_state,
    Detection,
    PRESENT,
    MISSING,
    UNKNOWN,
    COMPLIANT,
    NON_COMPLIANT,
    EventLogger,
)
from dashboard import compute_stats, read_events


class TestLabelClassification(unittest.TestCase):
    def test_normalize_label(self):
        self.assertEqual(normalize_label("Hard-Hat"), "hardhat")
        self.assertEqual(normalize_label("NO_Safety_Vest!"), "nosafetyvest")
        self.assertEqual(normalize_label("Person 01"), "person01")

    def test_classify_person(self):
        self.assertEqual(classify_label("person"), "person")
        self.assertEqual(classify_label("People"), "person")
        self.assertEqual(classify_label("worker"), "person")
        self.assertEqual(classify_label("human"), "person")

    def test_classify_helmet(self):
        self.assertEqual(classify_label("Hardhat"), "helmet_present")
        self.assertEqual(classify_label("helmet"), "helmet_present")
        self.assertEqual(classify_label("No-Hardhat"), "helmet_missing")
        self.assertEqual(classify_label("without_helmet"), "helmet_missing")
        self.assertEqual(classify_label("unworn helmet"), "helmet_missing")

    def test_classify_vest(self):
        self.assertEqual(classify_label("Safety Vest"), "vest_present")
        self.assertEqual(classify_label("reflective_vest"), "vest_present")
        self.assertEqual(classify_label("NO-Safety Vest"), "vest_missing")
        self.assertEqual(classify_label("missing vest"), "vest_missing")
        self.assertEqual(classify_label("notwearing_vest"), "vest_missing")

    def test_classify_other(self):
        self.assertEqual(classify_label("machinery"), "other")
        self.assertEqual(classify_label("cone"), "other")
        self.assertEqual(classify_label("vehicle"), "other")


class TestGeometryAndRegions(unittest.TestCase):
    def test_box_area(self):
        self.assertEqual(box_area((0, 0, 10, 20)), 200.0)
        self.assertEqual(box_area((10, 20, 10, 20)), 0.0)
        self.assertEqual(box_area((20, 30, 10, 10)), 0.0)

    def test_region_for_person_helmet(self):
        person = (0.0, 0.0, 100.0, 200.0)
        head_reg = region_for_person(person, "helmet")
        self.assertAlmostEqual(head_reg[0], 18.0)
        self.assertAlmostEqual(head_reg[1], 0.0)
        self.assertAlmostEqual(head_reg[2], 82.0)
        self.assertAlmostEqual(head_reg[3], 72.0)

    def test_region_for_person_vest(self):
        person = (0.0, 0.0, 100.0, 200.0)
        vest_reg = region_for_person(person, "vest")
        self.assertAlmostEqual(vest_reg[0], 10.0)
        self.assertAlmostEqual(vest_reg[1], 40.0)
        self.assertAlmostEqual(vest_reg[2], 90.0)
        self.assertAlmostEqual(vest_reg[3], 156.0)

    def test_overlap_fraction(self):
        box = (10, 10, 30, 30)  # area 400
        region = (0, 0, 20, 20)  # intersection 10x10 = 100
        self.assertAlmostEqual(overlap_fraction(box, region), 100.0 / 400.0)

    def test_center_in_region(self):
        region = (10, 10, 50, 50)
        self.assertTrue(center_in_region((20, 20, 30, 30), region))  # cx=25, cy=25
        self.assertFalse(center_in_region((0, 0, 10, 10), region))   # cx=5, cy=5


class TestPPEAssociation(unittest.TestCase):
    def test_associate_ppe_single_person(self):
        person = Detection(xyxy=(100, 100, 200, 400), label="person", confidence=0.9, role="person")
        helmet = Detection(xyxy=(130, 100, 170, 140), label="Hardhat", confidence=0.85, role="helmet_present")
        vest = Detection(xyxy=(120, 200, 180, 300), label="Safety Vest", confidence=0.88, role="vest_present")

        evidence = associate_ppe([person], [person, helmet, vest])
        self.assertEqual(len(evidence[0]["helmet"]), 1)
        self.assertEqual(evidence[0]["helmet"][0], (PRESENT, 0.85))
        self.assertEqual(len(evidence[0]["vest"]), 1)
        self.assertEqual(evidence[0]["vest"][0], (PRESENT, 0.88))

    def test_associate_ppe_two_workers(self):
        # Two workers side by side
        p1 = Detection(xyxy=(50, 100, 150, 400), label="person", confidence=0.9, role="person")
        p2 = Detection(xyxy=(250, 100, 350, 400), label="person", confidence=0.9, role="person")
        # Helmet on worker 2
        helmet_p2 = Detection(xyxy=(280, 100, 320, 140), label="Hardhat", confidence=0.85, role="helmet_present")

        evidence = associate_ppe([p1, p2], [p1, p2, helmet_p2])
        self.assertEqual(len(evidence[0]["helmet"]), 0)
        self.assertEqual(len(evidence[1]["helmet"]), 1)


class TestStatusEvaluation(unittest.TestCase):
    def test_frame_item_status(self):
        self.assertEqual(frame_item_status([]), UNKNOWN)
        self.assertEqual(frame_item_status([(PRESENT, 0.8)]), PRESENT)
        self.assertEqual(frame_item_status([(MISSING, 0.8)]), MISSING)

        # Conflicting with strong dominant confidence
        self.assertEqual(frame_item_status([(PRESENT, 0.9), (MISSING, 0.5)]), PRESENT)
        # Conflicting without clear margin
        self.assertEqual(frame_item_status([(PRESENT, 0.8), (MISSING, 0.7)]), UNKNOWN)

    def test_smooth_status(self):
        hist_compliant = deque([PRESENT, PRESENT, PRESENT, PRESENT], maxlen=8)
        self.assertEqual(smooth_status(hist_compliant, min_votes=3), PRESENT)

        hist_missing = deque([MISSING, MISSING, MISSING], maxlen=8)
        self.assertEqual(smooth_status(hist_missing, min_votes=3), MISSING)

        hist_mixed = deque([PRESENT, MISSING, PRESENT, MISSING], maxlen=8)
        self.assertEqual(smooth_status(hist_mixed, min_votes=3), UNKNOWN)

        hist_insufficient = deque([PRESENT, PRESENT], maxlen=8)
        self.assertEqual(smooth_status(hist_insufficient, min_votes=3), UNKNOWN)

    def test_overall_status(self):
        self.assertEqual(overall_status(PRESENT, PRESENT), COMPLIANT)
        self.assertEqual(overall_status(PRESENT, MISSING), NON_COMPLIANT)
        self.assertEqual(overall_status(MISSING, PRESENT), NON_COMPLIANT)
        self.assertEqual(overall_status(MISSING, MISSING), NON_COMPLIANT)
        self.assertEqual(overall_status(PRESENT, UNKNOWN), UNKNOWN)
        self.assertEqual(overall_status(UNKNOWN, PRESENT), UNKNOWN)
        self.assertEqual(overall_status(UNKNOWN, UNKNOWN), UNKNOWN)


class TestCrossingDetection(unittest.TestCase):
    def test_point_side_horizontal(self):
        # Line at 50% height (y=250 in 500h), hysteresis 10px
        # Above line (y < 240) -> -1
        self.assertEqual(point_side((100, 200), 500, 500, "horizontal", 0.5, 10), -1)
        # Below line (y > 260) -> 1
        self.assertEqual(point_side((100, 300), 500, 500, "horizontal", 0.5, 10), 1)
        # In deadband (y=255) -> 0
        self.assertEqual(point_side((100, 255), 500, 500, "horizontal", 0.5, 10), 0)

    def test_point_side_vertical(self):
        # Line at 50% width (x=250 in 500w), hysteresis 10px
        self.assertEqual(point_side((100, 200), 500, 500, "vertical", 0.5, 10), -1)
        self.assertEqual(point_side((300, 200), 500, 500, "vertical", 0.5, 10), 1)

    def test_check_zone_state(self):
        # 1000 width, left_pos=0.25 (250px), right_pos=0.75 (750px), hysteresis=10
        # Left walkway (< 240)
        self.assertEqual(check_zone_state((100, 300), 1000, 0.25, 0.75, 10), "LEFT_WALKWAY")
        # Right walkway (> 760)
        self.assertEqual(check_zone_state((900, 300), 1000, 0.25, 0.75, 10), "RIGHT_WALKWAY")
        # Center machine cell (260 - 740)
        self.assertEqual(check_zone_state((500, 300), 1000, 0.25, 0.75, 10), "MACHINE_ZONE")
        # In transition band near left border (245)
        self.assertEqual(check_zone_state((245, 300), 1000, 0.25, 0.75, 10), "TRANSITION")

class TestEventLoggerAndDashboard(unittest.TestCase):
    def test_event_logging_and_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_events.csv"
            logger = EventLogger(csv_path, snapshots=False)

            logger.log("0", 1, "PPE_COMPLIANT_CROSSING", "positive", PRESENT, PRESENT, "test_model", None)
            logger.log("0", 2, "PPE_NONCOMPLIANT_CROSSING", "positive", MISSING, PRESENT, "test_model", None)
            logger.log("1", 3, "PPE_UNKNOWN_AT_CROSSING", "positive", UNKNOWN, PRESENT, "test_model", None)

            events = read_events(csv_path)
            self.assertEqual(len(events), 3)

            stats = compute_stats(events)
            self.assertEqual(stats["total_crossings"], 3)
            self.assertEqual(stats["violations"], 1)
            self.assertEqual(stats["compliant"], 1)
            self.assertEqual(stats["unknown"], 1)
            self.assertIn("0", stats["cameras"])
            self.assertIn("1", stats["cameras"])
            self.assertEqual(stats["cameras"]["0"]["violations"], 1)

if __name__ == "__main__":
    unittest.main()
