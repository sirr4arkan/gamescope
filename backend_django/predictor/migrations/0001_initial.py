from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        migrations.CreateModel(
            name="PredictionLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("game_title", models.CharField(blank=True, max_length=200)),
                ("platform", models.CharField(max_length=32)),
                ("genre", models.CharField(max_length=32)),
                ("publisher", models.CharField(max_length=200)),
                ("release_year", models.PositiveSmallIntegerField()),
                ("predicted_global_sales", models.FloatField()),
            ],
            options={
                "db_table": "prediction_logs",
                "ordering": ["-created_at", "-id"],
            },
        ),
    ]
