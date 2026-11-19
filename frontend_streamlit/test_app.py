import base64
import json
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from ui import api_client
from ui.app_config import GroqSettings, SupabaseSettings
from ui.errors import ApiError


ROOT = Path(__file__).resolve().parent
METADATA = {
    "platforms": ["PS2", "PS3"],
    "genres": ["Action", "RPG"],
    "publishers": ["Nintendo", "Sega"],
    "year_range": [1980, 2016],
    "average_sales_by_genre": {"Action": 0.5, "RPG": 0.8},
    "average_sales_by_platform": {"PS2": 0.8, "PS3": 0.6},
    "sales_analysis": {
        "genre_analysis": {
            "category_name": "Genre",
            "categories": [
                {
                    "category": "RPG",
                    "average_global_sales": 0.8,
                    "observations": 2,
                }
            ],
            "total_observations": 4,
            "year_range": [1980, 2016],
        },
        "platform_analysis": {
            "category_name": "Platform",
            "categories": [
                {
                    "category": "PS2",
                    "average_global_sales": 0.8,
                    "observations": 3,
                }
            ],
            "total_observations": 4,
            "year_range": [1980, 2016],
        },
    },
    "dataset_summary": {
        "cleaned_rows": 16288,
        "mean_global_sales": 0.54,
        "median_global_sales": 0.17,
        "max_global_sales": 82.74,
    },
    "model_summary": {
        "selected_model": "LinearRegression",
        "training_rows": 13030,
        "test_rows": 3258,
        "train_test_split": 0.2,
        "raw_metrics": {"r2": 0.093313, "rmse": 1.892897, "mae": 0.559662},
        "clipped_metrics": {"r2": 0.095198, "rmse": 1.890928, "mae": 0.542939},
    },
}
PREDICTION = {
    "game_title": "Test game",
    "platform": "PS2",
    "genre": "Action",
    "publisher": "Nintendo",
    "release_year": 2016,
    "predicted_global_sales": 2.25,
    "category": "Standard",
    "created_at": "2025-01-01T10:00:00+00:00",
}
SUPABASE_SETTINGS = SupabaseSettings(
    url="https://gamescope-test.supabase.co",
    api_key="test-anon-key",
    use_bearer_token=True,
)


def _rendered_chart_count(node):
    count = int(type(node).__name__ == "UnknownElement")
    children = getattr(node, "children", {})
    if isinstance(children, dict):
        return count + sum(_rendered_chart_count(child) for child in children.values())
    return count


def _response(body=None, status=200):
    return SimpleNamespace(
        ok=200 <= status < 300,
        status_code=status,
        json=lambda: body,
    )


@contextmanager
def _mock_app_services(
    *,
    history=None,
    metadata=METADATA,
    metadata_error=None,
    predict_result=PREDICTION,
    predict_error=None,
    explanation="## Ringkasan\nAnalisis berdasarkan data aktual.",
    explanation_error=None,
):
    with ExitStack() as stack:
        metadata_mock = stack.enter_context(
            patch(
                "ui.api_client.get_metadata",
                side_effect=metadata_error,
                return_value=metadata,
            )
        )
        history_mock = stack.enter_context(
            patch(
                "ui.api_client.get_history",
                return_value=history if history is not None else [],
            )
        )
        predict_mock = stack.enter_context(
            patch(
                "ui.api_client.predict",
                side_effect=predict_error,
                return_value=predict_result,
            )
        )
        explanation_mock = stack.enter_context(
            patch(
                "ui.api_client.explain_market_analysis",
                side_effect=explanation_error,
                return_value=explanation,
            )
        )
        yield {
            "metadata": metadata_mock,
            "history": history_mock,
            "predict": predict_mock,
            "explanation": explanation_mock,
        }


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def test_local_artifacts_supply_metadata_and_prediction(self):
        from ui.local_inference import get_metadata, predict_locally

        metadata = get_metadata()
        self.assertEqual(metadata["dataset_summary"]["cleaned_rows"], 16288)
        self.assertEqual(metadata["year_range"], [1980, 2016])
        self.assertTrue(metadata["sales_analysis"]["genre_analysis"]["categories"])

        result = predict_locally(
            {
                "game_title": "  Test game  ",
                "platform": metadata["platforms"][0],
                "genre": metadata["genres"][0],
                "publisher": metadata["publishers"][0],
                "release_year": metadata["year_range"][1],
            }
        )
        self.assertEqual(result["game_title"], "Test game")
        self.assertGreaterEqual(result["predicted_global_sales"], 0)
        self.assertIn(result["category"], ("Niche", "Standard", "Blockbuster"))

    def test_prediction_validation_rejects_unknown_values(self):
        from ui.local_inference import predict_locally

        payload = {
            "game_title": "",
            "platform": "unknown platform",
            "genre": "Action",
            "publisher": "Nintendo",
            "release_year": 2016,
        }
        with self.assertRaisesRegex(ApiError, "Platform tidak tersedia"):
            predict_locally(payload)

    def test_prediction_facade_saves_local_result_to_history_service(self):
        prediction = dict(PREDICTION)
        with patch(
            "ui.api_client.predict_locally", return_value=prediction
        ) as infer, patch(
            "ui.api_client.insert_prediction",
            return_value="2025-01-01T10:00:00+00:00",
        ) as insert:
            result = api_client.predict({"platform": "PS2"})
        infer.assert_called_once_with({"platform": "PS2"})
        insert.assert_called_once_with(prediction)
        self.assertEqual(result["created_at"], "2025-01-01T10:00:00+00:00")
        self.assertEqual(result["category"], "Standard")

    def test_supabase_postgrest_reads_latest_history_and_inserts_only(self):
        from ui.supabase_rest import get_history, insert_prediction

        history_row = {
            "id": 31,
            "created_at": "2025-01-01T10:00:00+00:00",
            "game_title": "Existing game",
            "platform": "PS2",
            "genre": "Action",
            "publisher": "Nintendo",
            "release_year": 2016,
            "predicted_global_sales": 2.25,
        }
        with patch(
            "ui.supabase_rest.get_supabase_settings",
            return_value=SUPABASE_SETTINGS,
        ), patch(
            "ui.supabase_rest.requests.get",
            return_value=_response([history_row]),
        ) as get_request, patch(
            "ui.supabase_rest.requests.post",
            return_value=_response(status=201),
        ) as post_request:
            self.assertEqual(get_history(), [history_row])
            created_at = insert_prediction(PREDICTION)

        self.assertEqual(get_request.call_args.args[0], SUPABASE_SETTINGS.url + "/rest/v1/prediction_logs")
        self.assertEqual(
            get_request.call_args.kwargs["params"]["order"],
            "created_at.desc,id.desc",
        )
        self.assertEqual(get_request.call_args.kwargs["params"]["limit"], "20")
        self.assertEqual(post_request.call_count, 1)
        self.assertEqual(post_request.call_args.args[0], SUPABASE_SETTINGS.url + "/rest/v1/prediction_logs")
        sent_record = post_request.call_args.kwargs["json"]
        self.assertEqual(sent_record["game_title"], "Test game")
        self.assertEqual(sent_record["predicted_global_sales"], 2.25)
        self.assertEqual(sent_record["created_at"], created_at)
        self.assertNotIn("id", sent_record)
        self.assertEqual(
            post_request.call_args.kwargs["headers"]["Prefer"], "return=minimal"
        )

    def test_supabase_configuration_requires_https_and_rejects_service_key(self):
        from ui.app_config import get_supabase_settings

        with patch(
            "ui.app_config._get_secret",
            side_effect=lambda name: {
                "SUPABASE_URL": "http://gamescope-test.supabase.co",
                "SUPABASE_ANON_KEY": "public-key",
            }.get(name, ""),
        ):
            with self.assertRaisesRegex(ApiError, "HTTPS"):
                get_supabase_settings()

        claims = base64.urlsafe_b64encode(
            json.dumps({"role": "service_role"}).encode("utf-8")
        ).decode("ascii").rstrip("=")
        service_key = f"header.{claims}.signature"
        with patch(
            "ui.app_config._get_secret",
            side_effect=lambda name: {
                "SUPABASE_URL": "https://gamescope-test.supabase.co",
                "SUPABASE_ANON_KEY": service_key,
            }.get(name, ""),
        ):
            with self.assertRaisesRegex(ApiError, "service-role"):
                get_supabase_settings()

    def test_secrets_accept_publishable_key_and_report_missing_values(self):
        from ui.app_config import get_groq_settings, get_supabase_settings

        with patch(
            "ui.app_config._get_secret",
            side_effect=lambda name: {
                "SUPABASE_URL": "https://gamescope-test.supabase.co/",
                "SUPABASE_PUBLISHABLE_KEY": "sb_publishable_test",
            }.get(name, ""),
        ):
            settings = get_supabase_settings()
        self.assertEqual(settings.url, "https://gamescope-test.supabase.co")
        self.assertFalse(settings.use_bearer_token)

        with patch("ui.app_config._get_secret", return_value=""):
            with self.assertRaisesRegex(ApiError, "SUPABASE_URL"):
                get_supabase_settings()
            with self.assertRaisesRegex(ApiError, "GROQ_API_KEY"):
                get_groq_settings()

    def test_supabase_access_denial_is_informative_and_does_not_expose_response(self):
        from ui.supabase_rest import get_history

        response = _response({"message": "private internal response"}, status=403)
        with patch(
            "ui.supabase_rest.get_supabase_settings",
            return_value=SUPABASE_SETTINGS,
        ), patch("ui.supabase_rest.requests.get", return_value=response):
            with self.assertRaises(ApiError) as raised:
                get_history()
        self.assertIn("RLS", str(raised.exception))
        self.assertNotIn("private internal response", str(raised.exception))

    def test_history_is_refetched_instead_of_cached(self):
        from ui.supabase_rest import get_history

        row = {
            "id": 1,
            "created_at": "2025-01-01T10:00:00+00:00",
            "game_title": "Test",
            "platform": "PS2",
            "genre": "Action",
            "publisher": "Nintendo",
            "release_year": 2016,
            "predicted_global_sales": 1.0,
        }
        with patch(
            "ui.supabase_rest.get_supabase_settings",
            return_value=SUPABASE_SETTINGS,
        ), patch(
            "ui.supabase_rest.requests.get",
            side_effect=[_response([row]), _response([])],
        ) as get_request:
            self.assertEqual(get_history(), [row])
            self.assertEqual(get_history(), [])
        self.assertEqual(get_request.call_count, 2)

    def test_groq_explanation_uses_actual_context_and_safe_limits(self):
        from ui.groq_explainer import explain_market_analysis

        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="## Ringkasan\nData historis.")
                )
            ]
        )
        groq_factory = MagicMock()
        groq_factory.return_value.__enter__.return_value = client
        with patch(
            "ui.groq_explainer.get_metadata", return_value=METADATA
        ), patch(
            "ui.groq_explainer.get_groq_settings",
            return_value=GroqSettings("test-groq-key", "test-model"),
        ), patch("ui.groq_explainer.Groq", groq_factory):
            explanation = explain_market_analysis("genre_analysis")

        self.assertEqual(explanation, "## Ringkasan\nData historis.")
        call = client.chat.completions.create.call_args.kwargs
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["max_completion_tokens"], 450)
        self.assertIn("RPG", call["messages"][1]["content"])

    def test_home_displays_actual_local_summary_and_illustration(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        rendered = "\n".join(element.value for element in app.markdown)
        self.assertIn("GameScope", rendered)
        self.assertIn("16,288", rendered)
        self.assertIn("LinearRegression", rendered)
        self.assertFalse(app.warning)
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.form_submit_button)

    def test_prediction_page_submits_and_displays_result_without_backend(self):
        with _mock_app_services() as services:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/prediction.py").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.selectbox), 3)
            self.assertEqual(len(app.number_input), 1)
            self.assertEqual(app.number_input[0].value, 2016)
            app.text_input[0].set_value("Test game")
            app.form_submit_button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            services["predict"].assert_called_once()
            self.assertEqual(
                services["predict"].call_args.args[0]["release_year"], 2016
            )
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("2.250", rendered)
            self.assertIn("ESTIMASI PENJUALAN GLOBAL", rendered)
            self.assertIn("Standard", rendered)

    def test_market_charts_and_architecture_remain_available(self):
        with _mock_app_services():
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/analysis.py").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(_rendered_chart_count(app._tree.main), 2)
            self.assertEqual(
                [button.label for button in app.button],
                [
                    "Jelaskan Grafik Genre dengan AI",
                    "Jelaskan Grafik Platform dengan AI",
                ],
            )

            app.switch_page("pages/model.py").run(timeout=30)
            self.assertFalse(app.exception)
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("0.093", rendered)
            self.assertIn("1.893", rendered)
            self.assertIn("OneHotEncoder", rendered)
            self.assertIn("Supabase · Groq", rendered)
            self.assertIn("PostgREST · penjelasan AI", rendered)
            self.assertIn("Global Sales", rendered)
            self.assertNotIn("Django REST API", rendered)

    def test_model_diagrams_render_when_local_metadata_is_unavailable(self):
        with _mock_app_services(metadata_error=ApiError("artifacts tidak tersedia")):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/model.py").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            self.assertIn("Diagram tetap tersedia", app.warning[0].value)
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("workflow-diagram", rendered)
            self.assertIn("system-diagram", rendered)
            self.assertFalse(app.button)

    def test_analysis_explanations_are_click_only_and_regeneratable(self):
        explanations = {
            "genre_analysis": "## Genre\nAction merupakan agregat historis.",
            "platform_analysis": "## Platform\nPS2 merupakan agregat historis.",
        }
        with _mock_app_services() as services:
            services["explanation"].side_effect = (
                lambda analysis_type: explanations[analysis_type]
            )
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/analysis.py").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(_rendered_chart_count(app._tree.main), 2)
            self.assertEqual(len(app.button), 2)
            services["explanation"].assert_not_called()

            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(
                services["explanation"].call_args_list[-1].args[0],
                "genre_analysis",
            )
            self.assertEqual(
                app.session_state["genre_analysis_explanation"],
                explanations["genre_analysis"],
            )
            self.assertNotIn("platform_analysis_explanation", app.session_state)

            app.button[1].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(
                services["explanation"].call_args_list[-1].args[0],
                "platform_analysis",
            )
            self.assertEqual(
                app.session_state["platform_analysis_explanation"],
                explanations["platform_analysis"],
            )

            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(services["explanation"].call_count, 3)

    def test_analysis_service_error_keeps_charts_and_shows_safe_message(self):
        message = "Tambahkan GROQ_API_KEY ke Streamlit Secrets."
        with _mock_app_services(explanation_error=ApiError(message)):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/analysis.py").run(timeout=30)
            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn(message, [error.value for error in app.error])
            self.assertEqual(_rendered_chart_count(app._tree.main), 2)

    def test_explicit_light_theme_and_responsive_diagram_assets(self):
        config = ROOT.parent / ".streamlit" / "config.toml"
        theme = config.read_text(encoding="utf-8")
        self.assertIn('base = "light"', theme)
        self.assertIn('backgroundColor = "#F6F7F0"', theme)
        self.assertIn('primaryColor = "#456B4D"', theme)
        self.assertIn(
            'chartCategoricalColors = ["#456B4D", "#96AA83", "#C4D2B4"]',
            theme,
        )

        with _mock_app_services():
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/model.py").run(timeout=30)
            self.assertFalse(app.exception)
            diagrams = [
                element.value
                for element in app.markdown
                if "workflow-diagram" in element.value
                or "system-diagram" in element.value
            ]
            self.assertEqual(len(diagrams), 2)
            self.assertTrue(all('role="img"' in diagram for diagram in diagrams))
            self.assertTrue(all("is-animated" in diagram for diagram in diagrams))

            app.run(timeout=30)
            self.assertFalse(app.exception)
            diagrams = [
                element.value
                for element in app.markdown
                if "workflow-diagram" in element.value
                or "system-diagram" in element.value
            ]
            self.assertEqual(len(diagrams), 2)
            self.assertTrue(all("is-animated" not in diagram for diagram in diagrams))

        styles = (ROOT / "ui" / "styles.css").read_text(encoding="utf-8")
        self.assertIn("@keyframes diagram-enter", styles)
        self.assertIn("translate: -12px 0", styles)
        self.assertIn("@media (prefers-reduced-motion: reduce)", styles)

    def test_shared_sage_content_panel_and_home_process_strip(self):
        styles = (ROOT / "ui" / "styles.css").read_text(encoding="utf-8")
        self.assertIn("--ivory: #F6F7F0", styles)
        self.assertIn("--content-sage: #E9EFE2", styles)
        self.assertIn("--sage-soft: #F1F4EC", styles)
        self.assertIn('[data-testid="stMainBlockContainer"]', styles)
        self.assertIn("background: var(--content-sage)", styles)
        self.assertIn("@media (max-width: 760px)", styles)

        with _mock_app_services():
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            self.assertFalse(app.exception)
            rendered = "\n".join(element.value for element in app.markdown)
            for feature in (
                "Prediksi Penjualan Game",
                "Analisis Pasar Historis",
                "Wawasan Machine Learning",
                "Karakteristik Rilis",
                "Prediksi Machine Learning",
                "Estimasi Penjualan",
            ):
                self.assertIn(feature, rendered)
            self.assertIn("process-strip", rendered)

    def test_market_charts_use_light_background_and_green_palette(self):
        from ui.charts import COLORS, render_sales_chart

        with patch("ui.charts.st.altair_chart") as render:
            render_sales_chart({"Action": 0.5, "RPG": 0.8}, "Genre", height=280)
        chart = render.call_args.args[0].to_dict()
        self.assertEqual(chart["background"], "#FFFFFF")
        self.assertEqual(chart["config"]["view"]["fill"], "#FFFFFF")
        self.assertEqual(chart["config"]["axis"]["gridColor"], "#E9EFE2")
        self.assertEqual(
            chart["encoding"]["color"]["scale"]["range"],
            list(COLORS),
        )
        self.assertEqual(
            [item["field"] for item in chart["encoding"]["tooltip"]],
            ["Genre", "Sales"],
        )

    def test_guide_and_empty_history_render_without_fake_records(self):
        with _mock_app_services(history=[]):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/guide.py").run(timeout=30)
            self.assertFalse(app.exception)
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("Estimasi, bukan janji", rendered)

            app.switch_page("pages/history.py").run(timeout=30)
            self.assertFalse(app.exception)
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("BELUM ADA CATATAN", rendered)
            self.assertFalse(app.dataframe)

    def test_sales_units_remain_consistent_across_all_pages(self):
        with _mock_app_services():
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            self.assertFalse(app.exception)
            pages = (
                "pages/prediction.py",
                "pages/analysis.py",
                "pages/model.py",
                "pages/history.py",
                "pages/guide.py",
            )
            for page in pages:
                with self.subTest(page=page):
                    app.switch_page(page).run(timeout=30)
                    self.assertFalse(app.exception)
                    rendered = "\n".join(element.value for element in app.markdown)
                    self.assertNotIn("juta kopi", rendered.lower())
                    self.assertNotIn("copies", rendered.lower())

    def test_history_page_reads_supabase_rows(self):
        history = [
            {
                "id": 7,
                "created_at": "2025-01-01T10:00:00+00:00",
                "game_title": "Test game",
                "platform": "PS2",
                "genre": "Action",
                "publisher": "Nintendo",
                "release_year": 2016,
                "predicted_global_sales": 2.25,
            }
        ]
        with _mock_app_services(history=history):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/history.py").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 1)
            self.assertIn("Test game", app.dataframe[0].value.to_string())

    def test_history_page_reports_missing_supabase_configuration(self):
        message = "Tambahkan SUPABASE_URL ke Streamlit Secrets."
        with patch("ui.app_config._get_secret", return_value=""):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/history.py").run(timeout=30)
        self.assertFalse(app.exception)
        rendered = "\n".join(element.value for element in app.markdown)
        self.assertIn(message, rendered)

    def test_configuration_error_is_shown_instead_of_local_backend_hint(self):
        message = "Tambahkan SUPABASE_URL ke Streamlit Secrets."
        with _mock_app_services(metadata_error=ApiError("artifact tidak tersedia")):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            self.assertIn("Ringkasan dataset/model", app.warning[0].value)
            self.assertNotIn("aplikasi lokal sudah siap", app.warning[0].value)

        with _mock_app_services(predict_error=ApiError(message)):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/prediction.py").run(timeout=30)
            app.form_submit_button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn(message, [error.value for error in app.error])


if __name__ == "__main__":
    unittest.main()
