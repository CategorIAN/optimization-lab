from django.shortcuts import render


def home(request):
    """Render the Optimization Lab landing page."""
    return render(request, "visualizations/home.html")
