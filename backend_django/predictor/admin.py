from django.contrib import admin
from django.contrib.admin import AdminSite
from django.contrib.auth.admin import GroupAdmin, UserAdmin
from django.contrib.auth.models import Group, User

from predictor.models import PredictionLog


class GameScopeAdminSite(AdminSite):
    site_header = "GameScope — Admin Database"
    site_title = "GameScope Administration"
    index_title = "Kelola Database GameScope"

    def get_app_list(self, request, app_label=None):
        app_list = super().get_app_list(request, app_label)
        category_names = {
            "auth": "Authentication & Authorization",
            "predictor": "Prediction Management",
        }
        for app in app_list:
            app["name"] = category_names.get(app["app_label"], app["name"])
        return sorted(app_list, key=lambda app: app["name"].lower())


gamescope_admin_site = GameScopeAdminSite(name="admin")
gamescope_admin_site.register(User, UserAdmin)
gamescope_admin_site.register(Group, GroupAdmin)


@admin.register(PredictionLog, site=gamescope_admin_site)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "created_at",
        "game_title",
        "platform",
        "genre",
        "publisher",
        "release_year",
        "sales_in_millions",
    )
    list_display_links = ("id", "game_title")
    search_fields = ("game_title", "platform", "genre", "publisher")
    list_filter = ("platform", "genre", "publisher", "release_year")
    ordering = ("-created_at", "-id")
    date_hierarchy = "created_at"
    list_per_page = 25
    actions = ("delete_selected",)
    readonly_fields = ("id", "created_at")
    fields = (
        "id",
        "created_at",
        "game_title",
        "platform",
        "genre",
        "publisher",
        "release_year",
        "predicted_global_sales",
    )

    @admin.display(
        description="Predicted Global Sales (million units)",
        ordering="predicted_global_sales",
    )
    def sales_in_millions(self, obj):
        return f"{obj.predicted_global_sales:,.3f} million"

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        if db_field.name == "predicted_global_sales":
            kwargs["label"] = "Predicted Global Sales (million units)"
            kwargs["help_text"] = "Enter the estimate in millions of units."
        return super().formfield_for_dbfield(db_field, request, **kwargs)
