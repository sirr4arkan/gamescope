import streamlit as st


def render_ml_workflow(animate=False):
    animation_class = " is-animated" if animate else ""
    st.markdown(
        f"""
        <div class="diagram-frame workflow-diagram{animation_class}" role="img"
             aria-label="Alur Machine Learning dari Dataset melalui pembersihan data, pembagian data, Pipeline, Joblib, hingga estimasi Global Sales">
          <div class="workflow-flow">
            <div class="diagram-node workflow-source">
              <span class="diagram-index">01</span>
              <span class="diagram-kicker">DATASET</span>
              <strong>vgsales.csv</strong>
              <small>Data asli</small>
            </div>
            <span class="diagram-arrow" aria-hidden="true">→</span>
            <div class="diagram-node">
              <span class="diagram-index">02</span>
              <span class="diagram-kicker">PEMBERSIHAN DATA</span>
              <strong>Pembersihan data</strong>
              <small>Validasi kolom dan missing values</small>
            </div>
            <span class="diagram-arrow" aria-hidden="true">→</span>
            <div class="diagram-node">
              <span class="diagram-index">03</span>
              <span class="diagram-kicker">EVALUASI</span>
              <strong>Pembagian data latih-uji</strong>
              <small>80% data latih · 20% data uji</small>
            </div>
            <span class="diagram-arrow" aria-hidden="true">→</span>
            <div class="workflow-pipeline">
              <div class="pipeline-heading">
                <span class="diagram-index">04</span>
                <span class="diagram-kicker">PRA-PEMROSESAN + ESTIMATOR · DILATIH DENGAN DATA LATIH</span>
                <strong>Scikit-Learn Pipeline</strong>
              </div>
              <div class="pipeline-steps">
                <div class="pipeline-step">
                  <span class="diagram-kicker">COLUMNTRANSFORMER</span>
                  <strong>OneHotEncoder</strong>
                  <small>Platform · Genre · Publisher</small>
                  <small>Year: fitur numerik, tanpa penskalaan</small>
                </div>
                <span class="diagram-arrow pipeline-arrow" aria-hidden="true">→</span>
                <div class="pipeline-estimator">
                  <span class="diagram-kicker">ESTIMATOR</span>
                  <strong>Linear Regression</strong>
                </div>
              </div>
            </div>
            <span class="diagram-arrow" aria-hidden="true">→</span>
            <div class="diagram-node workflow-artifact">
              <span class="diagram-index">05</span>
              <span class="diagram-kicker">MODEL ARTIFACT</span>
              <strong>Joblib Pipeline</strong>
              <small>Pra-pemrosesan + estimator</small>
            </div>
            <span class="diagram-arrow" aria-hidden="true">→</span>
            <div class="diagram-node workflow-output">
              <span class="diagram-index">06</span>
              <span class="diagram-kicker">INFERENSI</span>
              <strong>Global Sales</strong>
              <small>Estimasi dalam juta unit terjual</small>
            </div>
          </div>
          <div class="diagram-caption">
            <span class="diagram-caption-dot"></span>
            Fitur: Platform, Genre, Year, Publisher
            <span class="diagram-caption-divider">·</span>
            Target: Global_Sales
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_architecture(animate=False):
    animation_class = " is-animated" if animate else ""
    st.markdown(
        f"""
        <div class="diagram-frame system-diagram{animation_class}" role="img"
             aria-label="Arsitektur: Pengguna memakai Streamlit, Streamlit berkomunikasi dengan Django REST API, Django memakai Pipeline Joblib dan SQLite untuk riwayat">
          <div class="system-flow">
            <div class="diagram-node system-user">
              <span class="diagram-index">01</span>
              <span class="diagram-kicker">PENGGUNA</span>
              <strong>Skenario game</strong>
              <small>Isi formulir dan baca hasil</small>
            </div>
            <div class="system-connector">
              <span class="system-connector-arrow">→</span>
              <small>masukan /<br>hasil</small>
            </div>
            <div class="diagram-node system-frontend">
              <span class="diagram-index">02</span>
              <span class="diagram-kicker">ANTARMUKA</span>
              <strong>Streamlit</strong>
              <small>Formulir · grafik · riwayat</small>
            </div>
            <div class="system-connector system-api-connector">
              <span class="system-connector-arrow">⇄</span>
              <small>GET metadata<br>POST predict<br>GET history</small>
            </div>
            <div class="diagram-node system-api">
              <span class="diagram-index">03</span>
              <span class="diagram-kicker">SISI SERVER</span>
              <strong>Django REST API</strong>
              <small>Validasi, inferensi, dan riwayat</small>
              <div class="database-link">
                <span>↕ &nbsp; Django ORM</span>
                <div class="database-card">
                  <strong>SQLite</strong>
                  <small>Simpan prediksi · baca riwayat</small>
                </div>
              </div>
            </div>
            <div class="system-connector">
              <span class="system-connector-arrow">⇄</span>
              <small>fitur<br>prediksi</small>
            </div>
            <div class="diagram-node system-model">
              <span class="diagram-index">04</span>
              <span class="diagram-kicker">INFERENSI</span>
              <strong>ML Pipeline</strong>
              <small>Joblib · dimuat sekali per proses</small>
            </div>
            <div class="system-connector">
              <span class="system-connector-arrow">→</span>
              <small>hasil<br>prediksi</small>
            </div>
            <div class="diagram-node system-output">
              <span class="diagram-index">05</span>
              <span class="diagram-kicker">HASIL API</span>
              <strong>Global Sales</strong>
              <small>Ditampilkan di Streamlit</small>
            </div>
          </div>
          <div class="diagram-caption">
            <span class="diagram-caption-dot"></span>
            Prediksi, metadata, dan riwayat melewati Django API
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
