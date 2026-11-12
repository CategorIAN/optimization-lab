from typing import Any
import json
import math
from time import perf_counter

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from dynamic_programming.segmented_least_squares import (
    best_line,
    fit_line,
    least_squares_error,
    segmented_least_squares,
)
from greedy.set_cover import greedy_set_cover
from .max_flow_trace import maximum_flow_trace
from .tsp_solver import solve_tsp

MAX_POINTS = 200
MAX_FLOW_NODES = 12
MAX_FLOW_EDGES = 36
MAX_FLOW_CAPACITY = 100
MAX_TSP_CITIES = 9


@ensure_csrf_cookie
def home(request: Any) -> Any:
    """Render the Optimization Lab landing page."""
    return render(request, "visualizations/home.html")


@ensure_csrf_cookie
def segmented_least_squares_page(request: Any) -> Any:
    """Render the interactive segmented least squares demonstration."""
    return render(request, "visualizations/segmented_least_squares.html")


def _invalid_request(message: Any) -> Any:
    """Execute ``_invalid_request``.

    Args:
        message: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    return JsonResponse({"error": message}, status=400)


@require_POST
def solve_segmented_least_squares(request: Any) -> Any:
    """Run the existing segmented least squares implementation for submitted points."""
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _invalid_request("Request body must be valid JSON.")

    points = payload.get("points") if isinstance(payload, dict) else None
    penalty = payload.get("penalty") if isinstance(payload, dict) else None

    if not isinstance(points, list):
        return _invalid_request("Points must be a list.")
    if len(points) < 2:
        return _invalid_request("At least two points are required.")
    if len(points) > MAX_POINTS:
        return _invalid_request(f"At most {MAX_POINTS} points are allowed.")

    parsed_points = []
    for point in points:
        if not isinstance(point, dict):
            return _invalid_request("Each point must contain numeric x and y values.")
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            return _invalid_request("Each point must contain numeric x and y values.")
        if not math.isfinite(x) or not math.isfinite(y):
            return _invalid_request("Point coordinates must be finite numbers.")
        parsed_points.append((x, y))

    try:
        penalty = float(penalty)
    except (TypeError, ValueError):
        return _invalid_request("Penalty must be a number.")
    if not math.isfinite(penalty) or penalty < 0:
        return _invalid_request("Penalty must be a finite number greater than or equal to zero.")

    ordered_points = sorted(parsed_points, key=lambda point: point[0])
    x_values = [point[0] for point in ordered_points]
    y_values = [point[1] for point in ordered_points]

    try:
        start_time = perf_counter()
        segments, objective = segmented_least_squares(x_values, y_values, penalty)
        execution_time_ms = (perf_counter() - start_time) * 1000
    except ValueError as error:
        return _invalid_request(str(error))

    fitted_segments = []
    total_squared_error = 0.0
    for start, end in segments:
        slope, intercept = best_line(x_values[start:end + 1], y_values[start:end + 1])
        squared_error = least_squares_error(x_values, y_values, start, end)
        total_squared_error += squared_error
        fitted_segments.append({
            "start_index": start,
            "end_index": end,
            "slope": slope,
            "intercept": intercept,
            "squared_error": squared_error,
            "start": {"x": x_values[start], "y": float(fit_line(x_values[start], slope, intercept))},
            "end": {"x": x_values[end], "y": float(fit_line(x_values[end], slope, intercept))},
        })

    return JsonResponse({
        "segments": fitted_segments,
        "segment_count": len(fitted_segments),
        "total_squared_error": total_squared_error,
        "objective": objective,
        "execution_time_ms": execution_time_ms,
    })


@ensure_csrf_cookie
def greedy_set_cover_page(request: Any) -> Any:
    """Render the interactive greedy set cover demonstration."""
    return render(request, "visualizations/greedy_set_cover.html")


@require_POST
def solve_greedy_set_cover(request: Any) -> Any:
    """Run the existing greedy set cover function for a browser-generated collection."""
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _invalid_request("Request body must be valid JSON.")

    collection = payload.get("collection") if isinstance(payload, dict) else None
    if not isinstance(collection, dict):
        return _invalid_request("Collection must be an object mapping subset IDs to element lists.")
    if len(collection) > 20:
        return _invalid_request("At most 20 subsets are allowed.")

    parsed_collection = {}
    all_elements = set()
    for subset_id, elements in collection.items():
        if not isinstance(subset_id, str) or not subset_id:
            return _invalid_request("Subset IDs must be non-empty strings.")
        if not isinstance(elements, list) or not all(isinstance(element, str) for element in elements):
            return _invalid_request("Each subset must contain a list of element IDs.")
        parsed_collection[subset_id] = set(elements)
        all_elements.update(elements)

    try:
        selected_subsets = greedy_set_cover(parsed_collection)
    except (TypeError, ValueError) as error:
        return _invalid_request(str(error))

    return JsonResponse({
        "selected_subsets": selected_subsets,
        "element_count": len(all_elements),
    })


@ensure_csrf_cookie
def maximum_flow_page(request: Any) -> Any:
    """Execute ``maximum_flow_page``.

    Args:
        request: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    return render(request, "visualizations/maximum_flow.html")


@require_POST
def trace_maximum_flow(request: Any) -> Any:
    """Execute ``trace_maximum_flow``.

    Args:
        request: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _invalid_request("Request body must be valid JSON.")

    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    edges = payload.get("edges") if isinstance(payload, dict) else None
    source = payload.get("source") if isinstance(payload, dict) else None
    sink = payload.get("sink") if isinstance(payload, dict) else None
    strategy = payload.get("strategy") if isinstance(payload, dict) else None
    if not isinstance(nodes, list) or not all(isinstance(node, str) and node for node in nodes):
        return _invalid_request("Nodes must be non-empty string IDs.")
    if len(nodes) > MAX_FLOW_NODES or len(set(nodes)) != len(nodes):
        return _invalid_request(f"Use at most {MAX_FLOW_NODES} unique nodes.")
    if source not in nodes or sink not in nodes or source == sink:
        return _invalid_request("Source and sink must be distinct existing nodes.")
    if strategy not in {"bfs", "dfs"}:
        return _invalid_request("Strategy must be 'bfs' or 'dfs'.")
    if not isinstance(edges, list) or len(edges) > MAX_FLOW_EDGES:
        return _invalid_request(f"Use at most {MAX_FLOW_EDGES} edges.")

    capacity = {}
    for edge in edges:
        if not isinstance(edge, dict):
            return _invalid_request("Each edge must be an object.")
        start, end, value = edge.get("source"), edge.get("target"), edge.get("value")
        if start not in nodes or end not in nodes or start == end:
            return _invalid_request("Edges must connect distinct existing nodes.")
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_FLOW_CAPACITY:
            return _invalid_request(f"Capacities must be integers from 0 to {MAX_FLOW_CAPACITY}.")
        if (start, end) in capacity:
            return _invalid_request("Duplicate directed edges are not allowed.")
        capacity[(start, end)] = value

    try:
        return JsonResponse(maximum_flow_trace(capacity, source, sink, strategy))
    except ValueError as error:
        return _invalid_request(str(error))


@ensure_csrf_cookie
def traveling_salesman_page(request: Any) -> Any:
    """Execute ``traveling_salesman_page``.

    Args:
        request: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    return render(request, "visualizations/traveling_salesman.html", {"max_cities": MAX_TSP_CITIES})


@require_POST
def solve_traveling_salesman(request: Any) -> Any:
    """Execute ``solve_traveling_salesman``.

    Args:
        request: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _invalid_request("Request body must be valid JSON.")
    points = payload.get("points") if isinstance(payload, dict) else None
    checkpoint = payload.get("checkpoint") if isinstance(payload, dict) else None
    if not isinstance(points, list) or not 2 <= len(points) <= MAX_TSP_CITIES:
        return _invalid_request(f"Use between 2 and {MAX_TSP_CITIES} cities.")
    parsed_points = []
    for point in points:
        if not isinstance(point, dict):
            return _invalid_request("Each city must have x and y coordinates.")
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            return _invalid_request("Each city must have numeric x and y coordinates.")
        if not math.isfinite(x) or not math.isfinite(y):
            return _invalid_request("City coordinates must be finite.")
        parsed_points.append((x, y))
    try:
        return JsonResponse(solve_tsp(parsed_points, checkpoint=checkpoint))
    except ValueError as error:
        return _invalid_request(str(error))
