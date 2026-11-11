from django.urls import path

from . import views

app_name = "visualizations"

urlpatterns = [
    path("", views.home, name="home"),
    path("segmented-least-squares/", views.segmented_least_squares_page, name="segmented_least_squares"),
    path("api/segmented-least-squares/solve/", views.solve_segmented_least_squares, name="solve_segmented_least_squares"),
    path("greedy-set-cover/", views.greedy_set_cover_page, name="greedy_set_cover"),
    path("api/greedy-set-cover/solve/", views.solve_greedy_set_cover, name="solve_greedy_set_cover"),
    path("maximum-flow/", views.maximum_flow_page, name="maximum_flow"),
    path("api/maximum-flow/trace/", views.trace_maximum_flow, name="trace_maximum_flow"),
    path("traveling-salesman/", views.traveling_salesman_page, name="traveling_salesman"),
    path("api/traveling-salesman/solve/", views.solve_traveling_salesman, name="solve_traveling_salesman"),
]
