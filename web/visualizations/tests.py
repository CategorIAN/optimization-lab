import json

from django.test import TestCase
from django.urls import reverse


class SegmentedLeastSquaresViewsTests(TestCase):
    def test_page_loads(self):
        response = self.client.get(reverse("visualizations:segmented_least_squares"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Segmented Least Squares")
        self.assertContains(response, "getScreenCTM")
        self.assertContains(response, "plot-click-target")

    def test_solve_endpoint_returns_fitted_segments_and_metrics(self):
        response = self.client.post(
            reverse("visualizations:solve_segmented_least_squares"),
            data=json.dumps({
                "points": [{"x": 2, "y": 5}, {"x": 0, "y": 1}, {"x": 1, "y": 3}],
                "penalty": 4,
            }),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(result["segments"][0]["start_index"], 0)
        self.assertEqual(result["segments"][0]["end_index"], 2)
        self.assertAlmostEqual(result["total_squared_error"], 0)
        self.assertAlmostEqual(result["objective"], 4)
        self.assertGreaterEqual(result["execution_time_ms"], 0)

    def test_solve_endpoint_rejects_invalid_input(self):
        response = self.client.post(
            reverse("visualizations:solve_segmented_least_squares"),
            data=json.dumps({"points": [{"x": 0, "y": 0}], "penalty": -1}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())


class GreedySetCoverViewsTests(TestCase):
    def test_page_loads(self):
        response = self.client.get(reverse("visualizations:greedy_set_cover"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Generate new diagram")
        self.assertContains(response, "pointInside")
        self.assertContains(response, "createSVGPoint")
        self.assertContains(response, "getScreenCTM")
        self.assertContains(response, "Add point")
        self.assertContains(response, "Add set")

    def test_solve_endpoint_uses_existing_greedy_selection_order(self):
        response = self.client.post(
            reverse("visualizations:solve_greedy_set_cover"),
            data=json.dumps({
                "collection": {
                    "S1": ["e1", "e2"],
                    "S2": ["e2", "e3"],
                    "S3": ["e3"],
                },
            }),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["selected_subsets"], ["S1", "S2"])
        self.assertEqual(response.json()["element_count"], 3)

    def test_solve_endpoint_handles_empty_collection(self):
        response = self.client.post(
            reverse("visualizations:solve_greedy_set_cover"),
            data=json.dumps({"collection": {}}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["selected_subsets"], [])
