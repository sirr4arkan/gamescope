import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from ui.api_client import get_history


ROOT = Path(__file__).resolve().parent
METADATA = {
    "platforms": ["PS2", "PS3"],
    "genres": ["Action", "RPG"],
    "publishers": ["Nintendo", "Sega"],
    "year_range": [1980, 2016],
    "average_sales_by_genre": {"Action": 0.5, "RPG": 0.8},
    "average_sales_by_platform": {"PS2": 0.8, "PS3": 0.6},
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


def _rendered_chart_count(node):
    count = int(type(node).__name__ == "UnknownElement")
    children = getattr(node, "children", {})
    if isinstance(children, dict):
        return count + sum(_rendered_chart_count(child) for child in children.values())
    return count


def _response(body, status=200):
    return SimpleNamespace(
        ok=200 <= status < 300,
        status_code=status,
        json=lambda: body,
    )


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()

    def tearDown(self):
        st.cache_data.clear()

    def _get(self, history=None, fail=False):
        def get(url, timeout):
            if fail:
                raise requests.ConnectionError("internal detail")
            if url.endswith("/metadata/"):
                return _response(METADATA)
            return _response({"history": history or []})

        return get

    def test_history_refetches_after_an_admin_deletion(self):
        responses = [
            _response({"history": [{"id": 1}]}),
            _response({"history": []}),
        ]
        with patch("requests.get", side_effect=responses) as get_request:
            self.assertEqual(get_history(), [{"id": 1}])
            self.assertEqual(get_history(), [])
        self.assertEqual(get_request.call_count, 2)

    def test_home_displays_actual_summary_and_local_illustration(self):
        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post"
        ) as post_request:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            self.assertFalse(app.exception)
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("GameScope", rendered)
            self.assertIn("16,288", rendered)
            self.assertIn("LinearRegression", rendered)
            self.assertFalse(app.warning)
            post_request.assert_not_called()
            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.form_submit_button)

    def test_prediction_page_submits_to_api_and_displays_result(self):
        history = []

        def post(url, json, timeout):
            history.append(
                {
                    "game_title": json["game_title"],
                    "platform": json["platform"],
                    "genre": json["genre"],
                    "publisher": json["publisher"],
                    "release_year": json["release_year"],
                    "predicted_global_sales": 2.25,
                    "created_at": "2025-01-01T10:00:00+00:00",
                }
            )
            return _response(
                {
                    **history[-1],
                    "id": 1,
                    "category": "Standard",
                }
            )

        with patch("requests.get", side_effect=self._get(history)), patch(
            "requests.post", side_effect=post
        ) as post_request:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/prediction.py").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.selectbox), 3)
            self.assertEqual(len(app.number_input), 1)
            self.assertEqual(app.number_input[0].value, 2016)
            app.text_input[0].set_value("Test game")
            app.form_submit_button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(post_request.call_count, 1)
            self.assertEqual(post_request.call_args.kwargs["json"]["release_year"], 2016)
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("2.250", rendered)
            self.assertIn("ESTIMASI PENJUALAN GLOBAL", rendered)
            self.assertIn("juta unit terjual", rendered)
            self.assertIn("Standard", rendered)

            app.switch_page("pages/history.py").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 1)
            self.assertIn("Test game", app.dataframe[0].value.to_string())

    def test_market_charts_have_ai_controls_and_model_page_remains_educational(self):
        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post"
        ) as post_request:
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
            post_request.assert_not_called()

            app.switch_page("pages/model.py").run(timeout=30)
            self.assertFalse(app.exception)
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("0.093", rendered)
            self.assertIn("1.893", rendered)
            self.assertIn("OneHotEncoder", rendered)
            self.assertIn("Platform · Genre · Publisher", rendered)
            self.assertIn("Django REST API", rendered)
            self.assertIn("Simpan prediksi · baca riwayat", rendered)
            self.assertIn("Global Sales", rendered)
            diagrams = [
                element.value
                for element in app.markdown
                if "workflow-diagram" in element.value or "system-diagram" in element.value
            ]
            self.assertEqual(len(diagrams), 2)
            self.assertFalse(app.button)

    def test_model_diagrams_and_buttons_render_when_metadata_fails(self):
        with patch("requests.get", side_effect=self._get(fail=True)), patch(
            "requests.post"
        ) as post_request:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/model.py").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            self.assertIn("Diagram tetap tersedia", app.warning[0].value)
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("workflow-diagram", rendered)
            self.assertIn("system-diagram", rendered)
            self.assertFalse(app.button)
            post_request.assert_not_called()

    def test_analysis_explanations_are_isolated_click_only_and_regeneratable(self):
        explanations = {
            "genre_analysis": "## Genre\nAction memiliki rata-rata tertinggi.",
            "platform_analysis": "## Platform\nPS2 memiliki rata-rata tertinggi.",
        }

        def post(url, json, timeout):
            return _response(
                {
                    "diagram_type": json["diagram_type"],
                    "explanation": explanations[json["diagram_type"]],
                }
            )

        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post", side_effect=post
        ) as post_request:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/analysis.py").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(_rendered_chart_count(app._tree.main), 2)
            self.assertEqual(len(app.button), 2)
            post_request.assert_not_called()

            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(post_request.call_count, 1)
            self.assertEqual(
                post_request.call_args.kwargs["json"],
                {"diagram_type": "genre_analysis"},
            )
            self.assertEqual(
                app.session_state["genre_analysis_explanation"],
                explanations["genre_analysis"],
            )
            self.assertNotIn(
                "platform_analysis_explanation",
                app.session_state,
            )

            app.button[1].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(post_request.call_count, 2)
            self.assertEqual(
                post_request.call_args.kwargs["json"],
                {"diagram_type": "platform_analysis"},
            )
            self.assertEqual(
                app.session_state["platform_analysis_explanation"],
                explanations["platform_analysis"],
            )

            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(post_request.call_count, 3)
            self.assertEqual(
                app.session_state["genre_analysis_explanation"],
                explanations["genre_analysis"],
            )

    def test_analysis_backend_failure_keeps_charts_and_shows_friendly_error(self):
        missing_key_message = (
            "Fitur penjelasan AI belum dikonfigurasi pada backend."
        )

        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post",
            return_value=_response({"error": missing_key_message}, status=503),
        ) as post_request:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/analysis.py").run(timeout=30)
            self.assertEqual(_rendered_chart_count(app._tree.main), 2)
            self.assertEqual(len(app.button), 2)
            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn(missing_key_message, [error.value for error in app.error])
            self.assertEqual(_rendered_chart_count(app._tree.main), 2)
            self.assertEqual(len(app.button), 2)
            app.button[1].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(post_request.call_count, 2)
            self.assertEqual(
                [call.kwargs["json"] for call in post_request.call_args_list],
                [
                    {"diagram_type": "genre_analysis"},
                    {"diagram_type": "platform_analysis"},
                ],
            )
            self.assertEqual(len(app.button), 2)

    def test_market_analysis_timeout_shows_friendly_error(self):
        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post", side_effect=requests.Timeout("private detail")
        ) as post_request:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/analysis.py").run(timeout=30)
            post_request.assert_not_called()
            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(_rendered_chart_count(app._tree.main), 2)
            self.assertTrue(app.error)
            self.assertIn("memerlukan waktu lebih lama", app.error[0].value)
            self.assertNotIn("private detail", app.error[0].value)

    def test_explicit_light_theme_and_responsive_diagram_assets(self):
        config = ROOT.parent / ".streamlit" / "config.toml"
        theme = config.read_text(encoding="utf-8")
        self.assertIn('base = "light"', theme)
        self.assertIn('backgroundColor = "#F6F7F0"', theme)
        self.assertIn('primaryColor = "#456B4D"', theme)
        self.assertIn('chartCategoricalColors = ["#456B4D", "#96AA83", "#C4D2B4"]', theme)

        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post"
        ):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/model.py").run(timeout=30)
            self.assertFalse(app.exception)
            diagrams = [
                element.value
                for element in app.markdown
                if "workflow-diagram" in element.value or "system-diagram" in element.value
            ]
            self.assertEqual(len(diagrams), 2)
            self.assertTrue(all('role="img"' in diagram for diagram in diagrams))
            self.assertTrue(all("is-animated" in diagram for diagram in diagrams))

            app.run(timeout=30)
            self.assertFalse(app.exception)
            diagrams = [
                element.value
                for element in app.markdown
                if "workflow-diagram" in element.value or "system-diagram" in element.value
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

        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post"
        ):
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
        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post"
        ):
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

    def test_sales_units_are_consistent_across_all_pages(self):
        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post"
        ):
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

    def test_backend_unavailable_shows_friendly_page_message(self):
        with patch("requests.get", side_effect=self._get(fail=True)), patch(
            "requests.post"
        ) as post_request:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            self.assertNotIn("internal detail", app.warning[0].value)

            app.switch_page("pages/prediction.py").run(timeout=30)
            self.assertFalse(app.exception)
            rendered = "\n".join(element.value for element in app.markdown)
            self.assertIn("Pilihan formulir", rendered)
            post_request.assert_not_called()

    def test_prediction_connection_failure_shows_friendly_error(self):
        with patch("requests.get", side_effect=self._get()), patch(
            "requests.post", side_effect=requests.ConnectionError("internal detail")
        ):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.switch_page("pages/prediction.py").run(timeout=30)
            app.form_submit_button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertIn("belum dapat diproses", app.error[0].value)
            self.assertNotIn("internal detail", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
