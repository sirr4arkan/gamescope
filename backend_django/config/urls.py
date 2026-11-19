from django.urls import path

from predictor.admin import gamescope_admin_site
from predictor import views


urlpatterns = [
    path("admin/", gamescope_admin_site.urls),
    path("api/predict/", views.predict, name="predict"),
    path("api/history/", views.history, name="history"),
    path("api/metadata/", views.metadata, name="metadata"),
    path("api/explain/", views.explain, name="explain"),
]
