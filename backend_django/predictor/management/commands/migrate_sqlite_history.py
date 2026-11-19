import math
import sqlite3
from datetime import timezone as datetime_timezone
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.core.management.color import no_style
from django.db import DatabaseError, connections, transaction
from django.utils.dateparse import parse_datetime
from django.utils import timezone

from predictor.models import PredictionLog


SOURCE_COLUMNS = (
    "id",
    "created_at",
    "game_title",
    "platform",
    "genre",
    "publisher",
    "release_year",
    "predicted_global_sales",
)


class Command(BaseCommand):
    help = "Import SQLite prediction history into the configured Django database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--source",
            default=str(Path(__file__).resolve().parents[3] / "db.sqlite3"),
            help="Path to the source SQLite database (default: backend_django/db.sqlite3).",
        )
        parser.add_argument("--database", default="default")

    def handle(self, *args, **options):
        source = Path(options["source"]).resolve()
        if not source.is_file():
            raise CommandError("The source SQLite database file does not exist.")

        try:
            rows = self._read_source(source)
        except (sqlite3.Error, OSError):
            raise CommandError(
                "Could not read prediction history from the source SQLite database."
            ) from None

        alias = options["database"]
        try:
            with transaction.atomic(using=alias):
                existing = PredictionLog.objects.using(alias).in_bulk(
                    [row["id"] for row in rows]
                )
                new_rows = []
                new_row_values = {}
                already_present = 0
                for row in rows:
                    current = existing.get(row["id"])
                    if current is None:
                        prediction = PredictionLog(**row)
                        new_rows.append(prediction)
                        new_row_values[row["id"]] = row
                    elif self._matches(current, row):
                        already_present += 1
                    else:
                        raise CommandError(
                            "A prediction ID already exists with different data; "
                            f"import stopped at ID {row['id']}."
                        )

                if new_rows:
                    PredictionLog.objects.using(alias).bulk_create(new_rows)
                    for prediction in new_rows:
                        prediction.created_at = new_row_values[prediction.pk]["created_at"]
                    PredictionLog.objects.using(alias).bulk_update(
                        new_rows, ["created_at"]
                    )
                    self._reset_sequence(alias)

                verified = PredictionLog.objects.using(alias).in_bulk(
                    [row["id"] for row in rows]
                )
                if len(verified) != len(rows) or any(
                    not self._matches(verified[row["id"]], row) for row in rows
                ):
                    raise CommandError(
                        "The imported prediction history did not pass verification."
                    )
                target_count = PredictionLog.objects.using(alias).count()
        except DatabaseError:
            raise CommandError(
                "Could not import prediction history. Verify the database connection "
                "and ensure migrations have been applied."
            ) from None

        self.stdout.write(
            self.style.SUCCESS(
                f"Source records: {len(rows)}; imported: {len(new_rows)}; "
                f"already present: {already_present}; "
                f"destination records: {target_count}."
            )
        )

    @staticmethod
    def _read_source(source):
        uri = source.as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as sqlite_connection:
            cursor = sqlite_connection.execute(
                "SELECT " + ", ".join(SOURCE_COLUMNS) + " FROM prediction_logs ORDER BY id"
            )
            rows = []
            for record in cursor.fetchall():
                row = dict(zip(SOURCE_COLUMNS, record))
                row["created_at"] = parse_datetime(row["created_at"])
                if row["created_at"] is None:
                    raise CommandError(
                        f"Invalid created_at value in source prediction ID {row['id']}."
                    )
                if timezone.is_naive(row["created_at"]):
                    row["created_at"] = timezone.make_aware(
                        row["created_at"], datetime_timezone.utc
                    )
                rows.append(row)
            return rows

    @staticmethod
    def _matches(existing, source):
        return (
            all(
                getattr(existing, field) == source[field]
                for field in SOURCE_COLUMNS
                if field not in {"created_at", "predicted_global_sales"}
            )
            and math.isclose(
                existing.predicted_global_sales,
                source["predicted_global_sales"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            and existing.created_at == source["created_at"]
        )

    @staticmethod
    def _reset_sequence(alias):
        connection = connections[alias]
        with connection.cursor() as cursor:
            for sql in connection.ops.sequence_reset_sql(
                no_style(), [PredictionLog]
            ):
                cursor.execute(sql)
