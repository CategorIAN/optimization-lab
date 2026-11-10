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

MAX_POINTS = 200


@ensure_csrf_cookie
def home(request):
    """Render the Optimization Lab landing page."""
    return render(request, "visualizations/home.html")


@ensure_csrf_cookie
def segmented_least_squares_page(request):
    """Render the interactive segmented least squares demonstration."""
    return render(request, "visualizations/segmented_least_squares.html")


def _invalid_request(message):
    return JsonResponse({"error": message}, status=400)


@require_POST
def solve_segmented_least_squares(request):
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
