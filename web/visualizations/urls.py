from django.urls import path

from . import views

app_name = "visualizations"

urlpatterns = [
    path("", views.home, name="home"),
    path("segmented-least-squares/", views.segmented_least_squares_page, name="segmented_least_squares"),
    path("api/segmented-least-squares/solve/", views.solve_segmented_least_squares, name="solve_segmented_least_squares"),
]
