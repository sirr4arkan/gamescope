from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError, connections, transaction

from predictor.models import PredictionLog


class Command(BaseCommand):
    help = "Check the configured database with a transactional insert and select."

    def add_arguments(self, parser):
        parser.add_argument("--database", default="default")

    def handle(self, *args, **options):
        alias = options["database"]
        connection = connections[alias]
        expected_columns = {
            "id",
            "created_at",
            "game_title",
            "platform",
            "genre",
            "publisher",
            "release_year",
            "predicted_global_sales",
        }
        try:
            with transaction.atomic(using=alias):
                connection.ensure_connection()
                with connection.cursor() as cursor:
                    description = connection.introspection.get_table_description(
                        cursor, PredictionLog._meta.db_table
                    )
                actual_columns = {column.name for column in description}
                if actual_columns != expected_columns:
                    raise CommandError(
                        "Database table prediction_logs does not match the Django model."
                    )

                probe = PredictionLog.objects.using(alias).create(
                    game_title="__database_check__",
                    platform="__check__",
                    genre="__check__",
                    publisher="__check__",
                    release_year=2000,
                    predicted_global_sales=0.0,
                )
                selected = PredictionLog.objects.using(alias).get(pk=probe.pk)
                if selected.game_title != "__database_check__":
                    raise CommandError("Database insert/select verification failed.")
                transaction.set_rollback(True, using=alias)
        except DatabaseError:
            raise CommandError(
                "Database check failed. Verify the configured environment variables, "
                "network access, and applied migrations."
            ) from None

        self.stdout.write(
            self.style.SUCCESS(
                f"Database check passed ({connection.vendor}); "
                "insert/select succeeded and the probe was rolled back."
            )
        )
        self.stdout.write("prediction_logs columns: " + ", ".join(sorted(actual_columns)))
