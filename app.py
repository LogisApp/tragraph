import streamlit as st
import tempfile
import os
import requests
from analysis_engine import TradingAnalysisEngine, LLM_PROVIDERS
from risk_engine import RiskConfig

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")


def is_ollama_available():
    """Verifica si Ollama está corriendo localmente."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


ollama_available = is_ollama_available()
available_providers = {
    k: v for k, v in LLM_PROVIDERS.items()
    if "Ollama" not in k or ollama_available
}

st.set_page_config(page_title="AI Trading Strategy Analyzer", layout="wide")

st.title("📊 AI Trading Strategy Comparator")
st.caption("Pipeline institucional: CLIP → LLM → Quant Engine → Risk Engine")

# --- Sidebar: configuración ---
with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("🤖 Proveedor LLM")
    selected_provider = st.selectbox(
        "Modelo de razonamiento",
        options=list(available_providers.keys()),
        index=0,
    )
    if not ollama_available:
        st.caption("ℹ️ Llama Vision no disponible (Ollama no detectado)")

    provider_info = LLM_PROVIDERS[selected_provider]
    key_env = provider_info.get("key_env")

    st.divider()
    st.subheader("🔑 API Keys")

    google_key = st.text_input(
        "Google API Key", type="password",
        value=os.getenv("GOOGLE_API_KEY", "")
    )
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key

    openai_key = st.text_input(
        "OpenAI API Key", type="password",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    st.divider()
    st.subheader("⚠️ Risk Management")
    account_balance = st.number_input("Balance de cuenta ($)", value=10000.0, step=1000.0)
    max_risk_trade = st.slider("Riesgo máx por trade (%)", 0.5, 5.0, 1.0, 0.5)
    max_daily_loss = st.slider("Pérdida máx diaria (%)", 1.0, 10.0, 3.0, 0.5)
    min_rr = st.slider("R:R mínimo", 1.0, 5.0, 2.0, 0.5)

    st.divider()
    frame_interval = st.slider(
        "Intervalo de frames", min_value=10, max_value=120,
        value=30, step=10,
    )

    has_required_key = True
    if key_env:
        has_required_key = bool(os.getenv(key_env, ""))
        if not has_required_key:
            st.warning(f"⚠️ {key_env} requerida para {selected_provider}")

# --- Inicializar engine ---
@st.cache_resource
def get_engine(_balance, _risk, _daily, _rr):
    config = RiskConfig(
        account_balance=_balance,
        max_risk_per_trade=_risk,
        max_daily_loss=_daily,
        min_rr_ratio=_rr,
    )
    return TradingAnalysisEngine(risk_config=config)

engine = get_engine(account_balance, max_risk_trade, max_daily_loss, min_rr)

# --- Tabs ---
tab_train, tab_analyze = st.tabs(["🧠 Entrenamiento", "📈 Análisis"])

# ==========================
# TAB: ENTRENAMIENTO
# ==========================
with tab_train:
    st.subheader("Entrenar con Videos de Estrategia")
    st.write(f"Carpeta de videos: `{VIDEOS_DIR}`")

    video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")]
    if video_files:
        st.write(f"**{len(video_files)} videos encontrados:**")
        for vf in sorted(video_files):
            size_mb = os.path.getsize(os.path.join(VIDEOS_DIR, vf)) / (1024 * 1024)
            st.text(f"  • {vf} ({size_mb:.1f} MB)")
    else:
        st.warning("No se encontraron videos MP4 en la carpeta `videos/`.")

    if engine.embeddings.is_trained():
        st.success("✅ Índice FAISS existente encontrado.")
    else:
        st.info("ℹ️ No hay índice entrenado. Haz clic en 'Entrenar' para crear uno.")

    if st.button("🚀 Entrenar", disabled=len(video_files) == 0):
        progress_bar = st.progress(0, text="Iniciando entrenamiento...")
        status_text = st.empty()

        def update_progress(msg):
            status_text.text(msg)

        try:
            result = engine.entrenar(
                videos_dir=VIDEOS_DIR,
                frame_interval=frame_interval,
                progress_callback=update_progress,
            )
            progress_bar.progress(100, text="¡Entrenamiento completado!")
            st.success("🎉 Entrenamiento exitoso")
            st.json(result)
        except Exception as e:
            st.error(f"Error durante entrenamiento: {e}")

# ==========================
# TAB: ANÁLISIS
# ==========================
with tab_analyze:
    st.subheader("Analizar Gráfico de Trading")
    st.info(f"🤖 Proveedor: **{selected_provider}** | 💰 Balance: **${account_balance:,.0f}** | ⚠️ Riesgo/trade: **{max_risk_trade}%**")

    uploaded_chart = st.file_uploader(
        "Sube el gráfico (PNG/JPG)", type=["png", "jpg", "jpeg"]
    )

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        can_analyze = engine.embeddings.is_trained() and has_required_key
        analyze_clicked = st.button("🔍 Analizar Pipeline Completo", disabled=not can_analyze)

    with col_btn2:
        execute_clicked = st.button("⚡ Ejecutar en MT5")

    if analyze_clicked:
        if not uploaded_chart:
            st.warning("Debe subir un gráfico.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_chart.read())
                chart_path = tmp.name

            try:
                # ════════════════════════════════════
                # PIPELINE COMPLETO
                # ════════════════════════════════════
                with st.spinner("Ejecutando pipeline institucional..."):
                    pipeline = engine.analizar_pipeline(
                        chart_path,
                        provider_name=selected_provider,
                    )
                    st.session_state["pipeline"] = pipeline
                    st.session_state["engine"] = engine

                # --- Resultado del pipeline ---
                st.divider()

                # Indicador visual de etapas
                stage_cols = st.columns(4)
                stage_cols[0].metric("1️⃣ CLIP", f"{pipeline.clip_confidence:.0f}%")
                stage_cols[1].metric("2️⃣ LLM", pipeline.strategy)
                stage_cols[2].metric("3️⃣ Quant", f"{pipeline.quant.quant_score:.0f}/100",
                                     delta="✅" if pipeline.quant.approved else "❌")
                stage_cols[3].metric("4️⃣ Risk", f"R:R {pipeline.quant.risk_reward_ratio:.1f}",
                                     delta="✅" if pipeline.risk.approved else "❌")

                st.divider()

                # Decisión final
                if pipeline.should_execute:
                    st.success(f"✅ **OPERACIÓN APROBADA** — Lotes: {pipeline.risk.position_size} | "
                               f"Riesgo: ${pipeline.risk.risk_amount:.2f} ({pipeline.risk.risk_percent:.1f}%)")
                else:
                    st.error(f"⛔ **OPERACIÓN RECHAZADA** — {pipeline.rejection_reason}")

                # Layout 2 columnas: info + gráfico
                res_left, res_right = st.columns([1, 1])

                with res_left:
                    st.subheader("Estrategia Recomendada")
                    st.write(f"**{pipeline.llm_response.estrategia_recomendada}**")

                    levels = pipeline.llm_response.puntos_de_entrada_salida
                    lcol1, lcol2, lcol3 = st.columns(3)
                    lcol1.metric("Entry", f"{levels.entry:.5f}")
                    lcol2.metric("Stop Loss", f"{levels.stop_loss:.5f}")
                    lcol3.metric("Take Profit", f"{levels.take_profit:.5f}")

                with res_right:
                    st.subheader("📊 Gráfico Analizado")
                    st.image(chart_path, use_container_width=True)

                # Detalles del pipeline en expanders
                with st.expander("🖼️ Patrones Similares (CLIP + FAISS)", expanded=False):
                    matches = pipeline.matches
                    if matches:
                        img_cols = st.columns(min(len(matches), 3))
                        for i, match in enumerate(matches[:6]):
                            col = img_cols[i % 3]
                            frame_path = match.get("path", "")
                            if os.path.exists(frame_path):
                                col.image(
                                    frame_path,
                                    caption=f"{match['video']}\nScore: {match['score']:.3f}",
                                    use_container_width=True,
                                )

                with st.expander("📐 Validación Cuantitativa", expanded=True):
                    st.code(pipeline.quant.details)
                    qcol1, qcol2 = st.columns(2)
                    qcol1.metric("Quant Score", f"{pipeline.quant.quant_score}/100")
                    qcol2.metric("R:R Ratio", f"{pipeline.quant.risk_reward_ratio:.2f}")

                with st.expander("🛡️ Risk Engine", expanded=True):
                    st.code(pipeline.risk.details)
                    rcol1, rcol2 = st.columns(2)
                    rcol1.metric("Position Size", f"{pipeline.risk.position_size} lotes")
                    rcol2.metric("Riesgo Restante Hoy", f"${pipeline.risk.daily_loss_remaining:.2f}")

                with st.expander("🤖 Razonamiento LLM", expanded=False):
                    st.write(pipeline.llm_response.razonamiento_tecnico)

            except Exception as e:
                st.error(str(e))
            finally:
                os.remove(chart_path)

    if not engine.embeddings.is_trained():
        st.caption("⚠️ Primero debes entrenar el modelo en la pestaña 'Entrenamiento'.")

    if execute_clicked:
        if "pipeline" not in st.session_state:
            st.warning("⚠️ Primero debes analizar un escenario.")
        else:
            pipeline = st.session_state["pipeline"]
            if not pipeline.should_execute:
                st.error(f"⛔ No se puede ejecutar: {pipeline.rejection_reason}")
            else:
                try:
                    result = st.session_state["engine"].ejecutar_si_valido(pipeline)
                    st.success("✅ Orden enviada correctamente")
                    st.json(str(result))
                except Exception as e:
                    st.error(str(e))
