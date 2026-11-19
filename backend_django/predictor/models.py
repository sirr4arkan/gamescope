from django.db import models


class PredictionLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    game_title = models.CharField(max_length=200, blank=True)
    platform = models.CharField(max_length=32)
    genre = models.CharField(max_length=32)
    publisher = models.CharField(max_length=200)
    release_year = models.PositiveSmallIntegerField()
    predicted_global_sales = models.FloatField()

    class Meta:
        db_table = "prediction_logs"
        ordering = ["-created_at", "-id"]
