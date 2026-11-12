# Optimization Lab

Optimization Lab is a Python project for implementing and comparing algorithms in combinatorial optimization, graph theory, and operations research. It grew out of my graduate coursework on the Traveling Salesman Problem and now covers problems in routing, covering, segmentation, and network flow.

## Algorithms and experiments

| Problem | Work in the lab |
| --- | --- |
| Traveling Salesman Problem | Compare a Deep Q-Learning approach with the exact Held–Karp dynamic programming algorithm on the same generated Euclidean graphs. |
| Set cover | Explore the greedy algorithm and visualize how the number of selected sets changes as the number of elements grows. |
| Segmented least squares | Explore how to divide ordered observations into segments by balancing line-fitting error against a penalty for each segment. |
| Maximum flow | Explore how edge capacities constrain the amount of flow a network can carry from a source to a sink. |

## Traveling Salesman Problem

The lab builds on my graduate Deep Q-Learning project. The experiment uses reusable Euclidean graph utilities to give both algorithms the same routing instances. Held–Karp computes an optimal tour, which provides a baseline for evaluating the tour length returned by the learned policy. The experiment displays the hyperparameters used for each training episode so the settings are visible alongside the results.

## Set cover

The greedy algorithm repeatedly chooses the set that covers the most elements still uncovered. The experiment plots the number of sets selected against the number of elements in the problem.

## Project direction

I’m building the lab to understand how optimization algorithms make decisions, where their trade-offs show up in experiments, and how to apply them to real problems in operations research and network analysis.
