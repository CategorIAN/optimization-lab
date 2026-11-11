import json
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from graphs.maximum_flow import augment_flow, ford_fulkerson
from visualizations.max_flow_trace import augmentation_operations, maximum_flow_trace
from visualizations.tsp_solver import solve_tsp


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


class MaximumFlowViewsTests(TestCase):
    capacity = {
        ("s", "a"): 4,
        ("s", "b"): 3,
        ("a", "b"): 2,
        ("b", "a"): 1,
        ("a", "t"): 3,
        ("b", "t"): 4,
    }

    def test_trace_agrees_with_ford_fulkerson_for_both_strategies(self):
        for strategy in ("bfs", "dfs"):
            trace = maximum_flow_trace(self.capacity, "s", "t", strategy)
            flow, value = ford_fulkerson(self.capacity, "s", "t", strategy)

            self.assertEqual(trace["maximum_flow"], value)
            self.assertEqual(
                {(edge["source"], edge["target"]): edge["value"] for edge in trace["final_flow"]},
                flow,
            )
            self.assertTrue(all("residual_before" in step for step in trace["steps"]))

    def test_trace_serializes_antiparallel_residual_capacity(self):
        trace = maximum_flow_trace({("s", "a"): 3, ("a", "s"): 2, ("a", "t"): 3}, "s", "t")

        self.assertTrue(all({"source", "target", "value"} <= set(edge) for edge in trace["initial"]["residual"]))

    def test_forward_capacity_is_used_before_reverse_cancellation(self):
        capacity = {("a", "b"): 2, ("b", "a"): 3}
        flow = {("a", "b"): 2, ("b", "a"): 0}

        operations = augmentation_operations(capacity, flow, ["b", "a"], 5)
        self.assertEqual(operations, [
            {"type": "increase", "source": "b", "target": "a", "value": 3},
            {"type": "cancel", "source": "a", "target": "b", "value": 2},
        ])
        self.assertEqual(augment_flow(capacity, flow, ["b", "a"]), {("a", "b"): 0, ("b", "a"): 3})

    def test_page_and_endpoint(self):
        page = self.client.get(reverse("visualizations:maximum_flow"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "BFS (Edmonds")
        self.assertContains(page, "DOMPoint")

        response = self.client.post(
            reverse("visualizations:trace_maximum_flow"),
            data=json.dumps({
                "nodes": ["s", "a", "t"],
                "edges": [
                    {"source": "s", "target": "a", "value": 2},
                    {"source": "a", "target": "t", "value": 2},
                ],
                "source": "s",
                "sink": "t",
                "strategy": "bfs",
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["maximum_flow"], 2)

    def test_endpoint_rejects_duplicate_edges_and_invalid_terminals(self):
        response = self.client.post(
            reverse("visualizations:trace_maximum_flow"),
            data=json.dumps({
                "nodes": ["s", "t"],
                "edges": [{"source": "s", "target": "t", "value": 1}, {"source": "s", "target": "t", "value": 2}],
                "source": "s",
                "sink": "s",
                "strategy": "bfs",
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)


class TravelingSalesmanViewsTests(TestCase):
    def test_solvers_return_closed_valid_tours_and_consistent_ratio(self):
        result = solve_tsp([(0, 0), (3, 0), (0, 4)])

        for key in ("q_learning_tour", "optimal_tour"):
            tour = result[key]
            self.assertEqual(len(tour), 4)
            self.assertEqual(tour[0], tour[-1])
            self.assertEqual(set(tour[:-1]), {0, 1, 2})
        self.assertAlmostEqual(
            result["approximation_ratio"],
            result["q_learning_length"] / result["optimal_length"],
        )

    def test_page_and_endpoint(self):
        page = self.client.get(reverse("visualizations:traveling_salesman"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Add graph")
        self.assertContains(page, "DOMPoint")

        response = self.client.post(
            reverse("visualizations:solve_traveling_salesman"),
            data=json.dumps({"points": [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 0, "y": 4}]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("optimal_tour", response.json())
        self.assertEqual(
            response.json()["hyperparameters"],
            {"p": 4, "T": 2, "eps": 0.01, "n": 3, "alpha": 0.1, "beta": 5, "gamma": 0.9},
        )

    @patch("visualizations.tsp_solver.TSP_RL")
    def test_default_solve_uses_final_theta_for_greedy_evaluation(self, model_class):
        from models.theta import ThetaObject

        final_theta = ThetaObject([[[1.0]] for _ in range(8)])
        model = model_class.return_value
        model.QLearning.return_value = ([], final_theta)
        model.calculateWalk.return_value = [0, 1, 2]

        response = self.client.post(
            reverse("visualizations:solve_traveling_salesman"),
            data=json.dumps({"points": [
                {"x": 10, "y": 10},
                {"x": 100, "y": 10},
                {"x": 10, "y": 100},
            ]}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        model.calculateWalk.assert_called_once_with(
            final_theta,
            model_class.call_args.args[0],
            exploreOption=False,
        )
        self.assertIn("Greedy tour generated after training", response.json()["training"])

    def test_endpoint_rejects_excessive_or_invalid_cities(self):
        response = self.client.post(
            reverse("visualizations:solve_traveling_salesman"),
            data=json.dumps({"points": [{"x": 0, "y": 0}]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
