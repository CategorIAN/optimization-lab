import unittest

import numpy as np

from graphs.complete_graph import CompleteGraph
from models.theta import ThetaObject
from traveling_salesman.tsp_rl import TSP_RL


class TestTSPRL(unittest.TestCase):
    def setUp(self):
        self.model = TSP_RL(
            p=1,
            T=1,
            eps=0,
            n=1,
            alpha=0.01,
            beta=1,
            gamma=1,
        )

        self.graph = CompleteGraph([
            [0, 2, 3],
            [2, 0, 4],
            [3, 4, 0],
        ])

        # With p=1, every parameter is a 1×1 array.
        # Ones make the network calculations easy to check by hand.
        self.theta = ThetaObject([
            np.ones((1, 1))
            for _ in range(8)
        ])

    def test_selected_vertex_indicator(self):
        np.testing.assert_array_equal(
            self.model.x(self.graph, [0, 2]),
            [1, 0, 1],
        )

    def test_reward_is_change_in_negative_tour_length(self):
        # [0, 1] closes to 0 → 1 → 0: distance 4.
        # [0, 1, 2] closes to 0 → 1 → 2 → 0: distance 9.
        self.assertEqual(
            self.model.c(self.graph, [0, 1]),
            -4,
        )
        self.assertEqual(
            self.model.r(self.graph, [0, 1], 2),
            -5,
        )

    def test_first_embedding_layer(self):
        layers = self.model.mu_list_final(
            self.theta, self.graph, [0]
        )

        self.assertEqual(len(layers), 2)

        np.testing.assert_array_equal(
            layers[0],
            [[0, 0, 0]],
        )

        # Initial embeddings are zero, so neighbor contributions are zero.
        #
        # Vertex 0: selected indicator 1 + distances 2 + 3 = 6
        # Vertex 1: selected indicator 0 + distances 2 + 4 = 6
        # Vertex 2: selected indicator 0 + distances 3 + 4 = 7
        np.testing.assert_allclose(
            layers[1],
            [[6, 6, 7]],
        )

    def test_q_vector(self):
        q_values, layers = self.model.Q_vec_mu_list(
            self.theta, self.graph, [0]
        )

        # Whole-graph score: 6 + 6 + 7 = 19.
        # Add each vertex's individual score:
        # [19 + 6, 19 + 6, 19 + 7].
        self.assertEqual(q_values.shape, (3,))
        np.testing.assert_allclose(
            q_values,
            [25, 25, 26],
        )
        self.assertEqual(len(layers), 2)

    def test_policy_selects_best_unvisited_vertex(self):
        self.assertEqual(
            self.model.vBest(self.theta, self.graph, [0]),
            2,
        )

        # Vertex 2 now has the highest score but is already visited.
        self.assertEqual(
            self.model.vBest(self.theta, self.graph, [0, 2]),
            1,
        )

    def test_completed_tour_has_no_available_action(self):
        tour = [0, 1, 2]

        self.assertIsNone(
            self.model.vBest(self.theta, self.graph, tour),
        )
        self.assertEqual(
            self.model.QBest(self.theta, self.graph, tour),
            0,
        )

    def test_calculate_walk_visits_each_vertex_once(self):
        tour = self.model.calculateWalk(
            self.theta, self.graph
        )

        self.assertEqual(sorted(tour), self.graph.vertices)


if __name__ == "__main__":
    unittest.main()