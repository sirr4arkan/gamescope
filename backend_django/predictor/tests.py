import json
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.db import DatabaseError
from django.test import TestCase, override_settings
from django.urls import reverse
import httpx
from groq import APIConnectionError, APITimeoutError, RateLimitError

from ml.train_model import build_pipeline, clean_dataset
from predictor.admin import PredictionLogAdmin, gamescope_admin_site
from predictor.models import PredictionLog
from predictor import views


class PreprocessingTests(TestCase):
    def test_success_categories_cover_threshold_boundaries(self):
        self.assertEqual(views._success_category(0.999), "Niche")
        self.assertEqual(views._success_category(1.0), "Standard")
        self.assertEqual(views._success_category(4.999), "Standard")
        self.assertEqual(views._success_category(5.0), "Blockbuster")

    def test_cleaning_removes_exact_duplicates_but_keeps_same_title_games(self):
        source = [
            {
                "Rank": 1,
                "Name": "Repeated",
                "Platform": "A",
                "Year": 2000,
                "Genre": "Action",
                "Publisher": "P",
                "Global_Sales": 1.0,
            },
            {
                "Rank": 2,
                "Name": "Repeated",
                "Platform": "B",
                "Year": 2001,
                "Genre": "Action",
                "Publisher": "P",
                "Global_Sales": 2.0,
            },
        ]
        from pandas import DataFrame

        cleaned, counts = clean_dataset(DataFrame(source + [source[0]]))
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(counts["identical_duplicates_removed"], 1)

    def test_cleaning_drops_invalid_numeric_and_negative_target_but_keeps_outlier(self):
        from pandas import DataFrame

        rows = [
            {
                "Platform": "A",
                "Genre": "Action",
                "Year": 2000,
                "Publisher": "P",
                "Global_Sales": 1000.0,
            },
            {
                "Platform": "A",
                "Genre": "Action",
                "Year": "invalid",
                "Publisher": "P",
                "Global_Sales": 1.0,
            },
            {
                "Platform": "A",
                "Genre": "Action",
                "Year": 2000,
                "Publisher": "P",
                "Global_Sales": -1.0,
            },
        ]
        cleaned, counts = clean_dataset(DataFrame(rows))
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["Global_Sales"], 1000.0)
        self.assertEqual(counts["invalid_numeric_values_removed"], 1)
        self.assertEqual(counts["invalid_or_negative_target_rows_removed"], 1)

    def test_pipeline_predicts_with_unseen_categories(self):
        from pandas import DataFrame, Series
        from sklearn.linear_model import LinearRegression

        training = DataFrame(
            {
                "Platform": ["A", "B", "A"],
                "Genre": ["Action", "RPG", "RPG"],
                "Year": [2000, 2001, 2002],
                "Publisher": ["P", "Q", "P"],
            }
        )
        pipeline = build_pipeline(LinearRegression()).fit(
            training, Series([1.0, 2.0, 1.5])
        )
        result = pipeline.predict(
            DataFrame(
                [{"Platform": "new", "Genre": "new", "Year": 2003, "Publisher": "new"}],
                columns=["Platform", "Genre", "Year", "Publisher"],
            )
        )
        self.assertEqual(len(result), 1)


class PredictionApiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        views._load_metadata.cache_clear()
        views._load_model.cache_clear()

    @classmethod
    def tearDownClass(cls):
        views._load_metadata.cache_clear()
        views._load_model.cache_clear()
        super().tearDownClass()

    def setUp(self):
        views._load_metadata.cache_clear()
        views._load_model.cache_clear()

    def test_metadata_contains_dynamic_choices_and_year_range(self):
        response = self.client.get("/api/metadata/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["platforms"])
        self.assertTrue(body["genres"])
        self.assertTrue(body["publishers"])
        self.assertEqual(len(body["year_range"]), 2)
        self.assertEqual(body["dataset_summary"]["cleaned_rows"], 16288)
        self.assertEqual(body["model_summary"]["selected_model"], "LinearRegression")
        self.assertEqual(
            body["model_summary"]["raw_metrics"]["r2"],
            0.09331303895320853,
        )
        for analysis_type, averages_key, counts_key in (
            (
                "genre_analysis",
                "average_sales_by_genre",
                "sales_observations_by_genre",
            ),
            (
                "platform_analysis",
                "average_sales_by_platform",
                "sales_observations_by_platform",
            ),
        ):
            analysis = body["sales_analysis"][analysis_type]
            self.assertEqual(
                analysis["total_observations"],
                body["dataset_summary"]["cleaned_rows"],
            )
            self.assertEqual(analysis["year_range"], body["year_range"])
            self.assertEqual(
                sum(item["observations"] for item in analysis["categories"]),
                analysis["total_observations"],
            )
            for item in analysis["categories"]:
                self.assertEqual(
                    item["average_global_sales"],
                    body[averages_key][item["category"]],
                )
                self.assertEqual(
                    item["observations"],
                    body[counts_key][item["category"]],
                )

    def test_saved_pipeline_loads_once_and_predicts_a_metadata_choice(self):
        from pandas import DataFrame

        model = views._load_model()
        self.assertIs(model, views._load_model())
        self.assertEqual(type(model.named_steps["estimator"]).__name__, "LinearRegression")
        choices = views._load_metadata()
        prediction = model.predict(
            DataFrame(
                [
                    {
                        "Platform": choices["platforms"][0],
                        "Genre": choices["genres"][0],
                        "Year": choices["year_range"][0],
                        "Publisher": choices["publishers"][0],
                    }
                ]
            )
        )
        self.assertEqual(len(prediction), 1)

    @patch("predictor.views._load_model")
    def test_prediction_is_clamped_logged_and_returned(self, load_model):
        class FakeModel:
            def predict(self, rows):
                self.rows = rows
                return [-2.0]

        model = FakeModel()
        load_model.return_value = model
        choices = views._load_metadata()
        response = self.client.post(
            "/api/predict/",
            data=json.dumps(
                {
                    "game_title": "Test game",
                    "platform": choices["platforms"][0],
                    "genre": choices["genres"][0],
                    "release_year": choices["year_range"][0],
                    "publisher": choices["publishers"][0],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_global_sales"], 0.0)
        self.assertEqual(response.json()["category"], "Niche")
        self.assertEqual(PredictionLog.objects.count(), 1)
        self.assertEqual(list(model.rows.columns), ["Platform", "Genre", "Year", "Publisher"])

    def test_invalid_input_returns_json_error_without_logging(self):
        response = self.client.post(
            "/api/predict/",
            data=json.dumps({"platform": "invalid"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertEqual(PredictionLog.objects.count(), 0)

    def test_unknown_categories_and_invalid_year_are_rejected(self):
        choices = views._load_metadata()
        valid_payload = {
            "platform": choices["platforms"][0],
            "genre": choices["genres"][0],
            "publisher": choices["publishers"][0],
            "release_year": choices["year_range"][1],
        }
        for field in ("platform", "genre", "publisher"):
            with self.subTest(category=field):
                payload = {**valid_payload, field: "unknown category"}
                response = self.client.post(
                    "/api/predict/",
                    data=json.dumps(payload),
                    content_type="application/json",
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(field, response.json()["error"])

        for invalid_year in (True, "2016", choices["year_range"][1] + 1):
            with self.subTest(release_year=invalid_year):
                response = self.client.post(
                    "/api/predict/",
                    data=json.dumps(
                        {
                            "platform": choices["platforms"][0],
                            "genre": choices["genres"][0],
                            "publisher": choices["publishers"][0],
                            "release_year": invalid_year,
                        }
                    ),
                    content_type="application/json",
                )
                self.assertEqual(response.status_code, 400)

    @patch("predictor.views._load_model", side_effect=FileNotFoundError("private path"))
    def test_unavailable_model_returns_safe_error(self, _load_model):
        choices = views._load_metadata()
        response = self.client.post(
            "/api/predict/",
            data=json.dumps(
                {
                    "platform": choices["platforms"][0],
                    "genre": choices["genres"][0],
                    "publisher": choices["publishers"][0],
                    "release_year": choices["year_range"][0],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(b"private path", response.content)
        self.assertEqual(PredictionLog.objects.count(), 0)

    @patch(
        "predictor.views.PredictionLog.objects.create",
        side_effect=DatabaseError("private database detail"),
    )
    def test_database_write_failure_returns_safe_error(self, _create):
        choices = views._load_metadata()
        response = self.client.post(
            "/api/predict/",
            data=json.dumps(
                {
                    "platform": choices["platforms"][0],
                    "genre": choices["genres"][0],
                    "publisher": choices["publishers"][0],
                    "release_year": choices["year_range"][0],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(b"private database detail", response.content)

    def test_history_is_empty_when_no_predictions_exist(self):
        response = self.client.get("/api/history/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"history": []})

    @patch(
        "predictor.views.PredictionLog.objects.all",
        side_effect=DatabaseError("private database detail"),
    )
    def test_history_database_failure_returns_safe_error(self, _all):
        response = self.client.get("/api/history/")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(b"private database detail", response.content)

    def test_history_returns_newest_at_most_twenty_records(self):
        for index in range(22):
            PredictionLog.objects.create(
                game_title=str(index),
                platform="platform",
                genre="genre",
                publisher="publisher",
                release_year=2000,
                predicted_global_sales=float(index),
            )
        response = self.client.get("/api/history/")
        self.assertEqual(response.status_code, 200)
        body = response.json()["history"]
        self.assertEqual(len(body), 20)
        self.assertEqual(body[0]["game_title"], "21")


class AIExplanationApiTests(TestCase):
    def _post(self, analysis_type, extra=None):
        payload = {"diagram_type": analysis_type}
        if extra:
            payload.update(extra)
        return self.client.post(
            "/api/explain/",
            data=json.dumps(payload),
            content_type="application/json",
        )

    @override_settings(
        GROQ_API_KEY="test-key",
        GROQ_MODEL="openai/gpt-oss-20b",
    )
    @patch("predictor.ai_explainer.Groq")
    def test_valid_chart_analyses_use_dataset_context_and_return_markdown(
        self, groq_client
    ):
        client = groq_client.return_value.__enter__.return_value
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="## Ringkasan\nAnalisis historis.")
                )
            ]
        )

        for analysis_type in ("genre_analysis", "platform_analysis"):
            with self.subTest(analysis_type=analysis_type):
                response = self._post(analysis_type)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    {
                        "diagram_type": analysis_type,
                        "explanation": "## Ringkasan\nAnalisis historis.",
                    },
                )

        self.assertEqual(groq_client.call_count, 2)
        metadata = views._load_metadata()
        calls = client.chat.completions.create.call_args_list
        genre_prompt = calls[0].kwargs["messages"][1]["content"]
        platform_prompt = calls[1].kwargs["messages"][1]["content"]
        self.assertIn(
            json.dumps(
                metadata["sales_analysis"]["genre_analysis"],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            genre_prompt,
        )
        self.assertIn(
            json.dumps(
                metadata["sales_analysis"]["platform_analysis"],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            platform_prompt,
        )
        self.assertIn("kategori tertinggi dan terendah", calls[0].kwargs["messages"][0]["content"])
        self.assertIn(
            "jangan membuat prediksi masa depan",
            calls[1].kwargs["messages"][0]["content"],
        )
        self.assertIn("Genre", genre_prompt)
        self.assertIn("Platform", platform_prompt)
        self.assertIn("juta unit terjual", genre_prompt)

    @override_settings(GROQ_API_KEY="test-key")
    @patch("predictor.ai_explainer.Groq")
    def test_unknown_analysis_extra_prompt_and_wrong_method_are_rejected(
        self, groq_client
    ):
        self.assertEqual(self._post("unknown").status_code, 400)
        self.assertEqual(
            self._post("genre_analysis", {"prompt": "arbitrary prompt"}).status_code,
            400,
        )
        self.assertEqual(self.client.get("/api/explain/").status_code, 405)
        groq_client.assert_not_called()

    @override_settings(GROQ_API_KEY="")
    @patch("predictor.ai_explainer.Groq")
    def test_missing_api_key_returns_safe_error_without_provider_call(self, groq_client):
        response = self._post("genre_analysis")
        self.assertEqual(response.status_code, 503)
        self.assertIn("belum dikonfigurasi", response.json()["error"])
        groq_client.assert_not_called()

    @override_settings(GROQ_API_KEY="test-key")
    @patch("predictor.views._load_metadata", side_effect=OSError("private path"))
    @patch("predictor.ai_explainer.Groq")
    def test_dataset_failure_returns_safe_error_without_provider_call(
        self, groq_client, _load_metadata
    ):
        response = self._post("platform_analysis")
        self.assertEqual(response.status_code, 503)
        self.assertIn("Data analisis", response.json()["error"])
        self.assertNotIn("private path", response.content.decode())
        groq_client.assert_not_called()

    @override_settings(GROQ_API_KEY="test-key")
    @patch("predictor.ai_explainer.Groq")
    def test_provider_failures_are_safe_and_status_specific(self, groq_client):
        request = httpx.Request(
            "POST", "https://api.groq.com/openai/v1/chat/completions"
        )
        rate_limit = RateLimitError(
            "private provider detail",
            response=httpx.Response(429, request=request),
            body={"detail": "private provider detail"},
        )
        failures = (
            (rate_limit, 429, "sedang sibuk"),
            (APITimeoutError(request=request), 504, "waktu lebih lama"),
            (APIConnectionError(request=request), 502, "belum dapat dibuat"),
        )
        client = groq_client.return_value.__enter__.return_value

        for failure, status_code, message in failures:
            with self.subTest(status_code=status_code):
                client.chat.completions.create.side_effect = failure
                response = self._post("platform_analysis")
                self.assertEqual(response.status_code, status_code)
                self.assertIn(message, response.json()["error"])
                self.assertNotIn(b"private provider detail", response.content)

    @override_settings(GROQ_API_KEY="test-key")
    @patch("predictor.ai_explainer.Groq")
    def test_empty_or_invalid_provider_response_returns_safe_error(self, groq_client):
        client = groq_client.return_value.__enter__.return_value
        invalid_responses = (
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=None),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="  "))]
            ),
        )
        for invalid_response in invalid_responses:
            with self.subTest(invalid_response=invalid_response):
                client.chat.completions.create.return_value = invalid_response
                response = self._post("genre_analysis")
                self.assertEqual(response.status_code, 502)
                self.assertIn("tidak dapat digunakan", response.json()["error"])


class DjangoAdminTests(TestCase):
    def test_admin_requires_login_and_denies_non_staff_users(self):
        response = self.client.get("/admin/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response.url)

        user = get_user_model().objects.create_user(
            username="ordinary-user",
            password="test-password",
        )
        self.client.force_login(user)
        response = self.client.get("/admin/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response.url)

    def test_superuser_sees_branding_and_model_categories(self):
        user = get_user_model().objects.create_superuser(
            username="admin-test",
            email="admin-test@example.test",
            password="test-password",
        )
        self.client.force_login(user)

        response = self.client.get("/admin/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "GameScope")
        self.assertContains(response, "Admin Database")
        self.assertContains(response, "GameScope Administration")
        self.assertContains(response, "Kelola Database GameScope")
        self.assertContains(response, "Prediction Management")
        self.assertContains(response, "Authentication &amp; Authorization")
        self.assertContains(response, "Prediction logs")
        self.assertContains(response, "/static/admin/css/gamescope_admin.")

    def test_staff_needs_model_permission_to_view_prediction_logs(self):
        user = get_user_model().objects.create_user(
            username="prediction-viewer",
            password="test-password",
            is_staff=True,
        )
        self.client.force_login(user)

        response = self.client.get("/admin/")
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Prediction Management")

        permission = Permission.objects.get(
            content_type__app_label="predictor",
            codename="view_predictionlog",
        )
        user.user_permissions.add(permission)
        log = PredictionLog.objects.create(
            game_title="Admin detail test",
            platform="platform",
            genre="genre",
            publisher="publisher",
            release_year=2000,
            predicted_global_sales=1.25,
        )

        response = self.client.get("/admin/")
        self.assertContains(response, "Prediction Management")
        detail = self.client.get(
            reverse("admin:predictor_predictionlog_change", args=[log.pk])
        )
        self.assertEqual(detail.status_code, 200)
        self.assertContains(detail, "Admin detail test")

        edit_attempt = self.client.post(
            reverse("admin:predictor_predictionlog_change", args=[log.pk]),
            {"game_title": "Unauthorized edit"},
        )
        self.assertEqual(edit_attempt.status_code, 403)
        add_attempt = self.client.post(
            reverse("admin:predictor_predictionlog_add"),
            {"game_title": "Unauthorized add"},
        )
        self.assertEqual(add_attempt.status_code, 403)
        log.refresh_from_db()
        self.assertEqual(log.game_title, "Admin detail test")

    def test_prediction_admin_supports_search_filters_and_paging(self):
        model_admin = gamescope_admin_site._registry[PredictionLog]

        self.assertEqual(
            model_admin.search_fields,
            ("game_title", "platform", "genre", "publisher"),
        )
        self.assertEqual(
            model_admin.list_filter,
            ("platform", "genre", "publisher", "release_year"),
        )
        self.assertEqual(model_admin.list_per_page, 25)
        self.assertEqual(model_admin.ordering, ("-created_at", "-id"))


class PredictionLogDeletionAdminTests(TestCase):
    def setUp(self):
        user = get_user_model().objects.create_superuser(
            username="delete-admin",
            email="delete-admin@example.test",
            password="test-password",
        )
        self.client.force_login(user)

    def _create_log(self, title):
        return PredictionLog.objects.create(
            game_title=title,
            platform="delete-platform",
            genre="delete-genre",
            publisher="delete-publisher",
            release_year=2000,
            predicted_global_sales=1.5,
        )

    def test_single_delete_requires_confirmation_and_removes_history_entry(self):
        log = self._create_log("Single deletion test")
        change_url = reverse(
            "admin:predictor_predictionlog_change", args=[log.pk]
        )
        delete_url = reverse(
            "admin:predictor_predictionlog_delete", args=[log.pk]
        )

        detail = self.client.get(change_url)
        self.assertEqual(detail.status_code, 200)
        self.assertContains(detail, "Delete")
        confirmation = self.client.get(delete_url)
        self.assertEqual(confirmation.status_code, 200)
        self.assertContains(confirmation, "Prediction logs: 1")

        response = self.client.post(delete_url, {"post": "yes"})

        self.assertEqual(response.status_code, 302)
        self.assertFalse(PredictionLog.objects.filter(pk=log.pk).exists())
        history_ids = [
            item["id"] for item in self.client.get("/api/history/").json()["history"]
        ]
        self.assertNotIn(log.pk, history_ids)

    def test_bulk_delete_shows_count_and_requires_confirmation(self):
        staff = get_user_model().objects.create_user(
            username="bulk-delete-staff",
            is_staff=True,
        )
        delete_permissions = Permission.objects.filter(
            content_type__app_label="predictor",
            codename__in=("view_predictionlog", "delete_predictionlog"),
        )
        staff.user_permissions.set(delete_permissions)
        self.client.force_login(staff)
        logs = [
            self._create_log("Bulk deletion test one"),
            self._create_log("Bulk deletion test two"),
        ]
        log_ids = [str(log.pk) for log in logs]
        changelist_url = reverse("admin:predictor_predictionlog_changelist")
        list_page = self.client.get(changelist_url)
        self.assertContains(list_page, 'name="_selected_action"')
        self.assertContains(list_page, 'value="delete_selected"')
        action_data = {
            "action": "delete_selected",
            "_selected_action": log_ids,
        }

        confirmation = self.client.post(changelist_url, action_data)

        self.assertEqual(confirmation.status_code, 200)
        self.assertContains(confirmation, "Prediction logs: 2")
        self.assertEqual(PredictionLog.objects.filter(pk__in=log_ids).count(), 2)

        response = self.client.post(
            changelist_url, {**action_data, "post": "yes"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(PredictionLog.objects.filter(pk__in=log_ids).count(), 0)
        history_ids = {
            item["id"] for item in self.client.get("/api/history/").json()["history"]
        }
        self.assertTrue(history_ids.isdisjoint(int(log_id) for log_id in log_ids))

    def test_bulk_delete_selection_cancel_and_confirmation_preserve_unselected_rows(self):
        logs = [
            self._create_log(f"Bulk selection test {index}")
            for index in range(6)
        ]
        selected_ids = [str(log.pk) for log in logs[:2]]
        all_ids = {log.pk for log in logs}
        unselected_ids = all_ids - {int(log_id) for log_id in selected_ids}
        changelist_url = reverse("admin:predictor_predictionlog_changelist")
        action_data = {
            "action": "delete_selected",
            "_selected_action": selected_ids,
        }

        confirmation = self.client.post(changelist_url, action_data)

        self.assertEqual(confirmation.status_code, 200)
        self.assertContains(confirmation, "Prediction logs: 2")
        self.assertEqual(PredictionLog.objects.filter(pk__in=all_ids).count(), 6)

        self.client.get(changelist_url)
        self.assertEqual(PredictionLog.objects.filter(pk__in=all_ids).count(), 6)

        confirmation = self.client.post(changelist_url, action_data)
        self.assertEqual(confirmation.status_code, 200)
        response = self.client.post(
            changelist_url, {**action_data, "post": "yes"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(PredictionLog.objects.filter(pk__in=all_ids).count(), 4)
        self.assertFalse(
            PredictionLog.objects.filter(pk__in=selected_ids).exists()
        )
        self.assertEqual(
            set(
                PredictionLog.objects.filter(pk__in=unselected_ids)
                .values_list("pk", flat=True)
            ),
            unselected_ids,
        )

    def test_staff_without_delete_permission_cannot_delete_records(self):
        staff = get_user_model().objects.create_user(
            username="delete-viewer",
            is_staff=True,
        )
        view_permission = Permission.objects.get(
            content_type__app_label="predictor",
            codename="view_predictionlog",
        )
        staff.user_permissions.add(view_permission)
        self.client.force_login(staff)
        log = self._create_log("Permission deletion test")
        delete_url = reverse(
            "admin:predictor_predictionlog_delete", args=[log.pk]
        )

        response = self.client.get(delete_url)

        self.assertEqual(response.status_code, 403)
        list_page = self.client.get(
            reverse("admin:predictor_predictionlog_changelist")
        )
        self.assertNotContains(list_page, 'value="delete_selected"')
        self.assertTrue(PredictionLog.objects.filter(pk=log.pk).exists())
