import streamlit as st
import pandas as pd
import joblib
import torch
import seaborn as sns
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from transformers import AutoModel, AutoTokenizer, Trainer
import time
import sys

sys.path.append("src")
from ollama_llm.inference import predict_llm_batch

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Korean Restaurant ABSA",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

/* Aspect pills */
.pill-food     { background:#fef9c3; color:#854d0e; padding:0.2rem 0.7rem; border-radius:14px; font-size:0.8rem; font-weight:600; }
.pill-price    { background:#dcfce7; color:#166534; padding:0.2rem 0.7rem; border-radius:14px; font-size:0.8rem; font-weight:600; }
.pill-service  { background:#dbeafe; color:#1e40af; padding:0.2rem 0.7rem; border-radius:14px; font-size:0.8rem; font-weight:600; }
.pill-ambience { background:#f3e8ff; color:#6b21a8; padding:0.2rem 0.7rem; border-radius:14px; font-size:0.8rem; font-weight:600; }

/* Sentiment badges */
.sent-pos  { background:#dcfce7; color:#166534; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.82rem; font-weight:600; }
.sent-neg  { background:#fee2e2; color:#991b1b; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.82rem; font-weight:600; }
.sent-none { background:#f1f5f9; color:#475569; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.82rem; font-weight:600; }

/* Mono font for numbers/times */
.mono { font-family:'JetBrains Mono', monospace; font-size:0.85rem; color:#0369a1; }
</style>
""", unsafe_allow_html=True)

st.title("From TF-IDF to LLMs: Aspect-Based Sentiment Analysis on Korean Restaurant Reviews")
st.markdown(
    "Compare three models — **TF-IDF + LR**, **KcELECTRA**, and **Qwen 2.5** — "
    "on aspect-level sentiment across **FOOD**, **PRICE**, **SERVICE**, and **AMBIENCE**."
)
st.divider()

# ─── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    combined = pd.read_csv("results/overall_metrics.csv")
    per_aspect = pd.read_csv("results/aspect_metrics.csv")
    return combined, per_aspect

combined, per_aspect = load_results()

# ─── Load Models ───────────────────────────────────────────────────────────────
from evaluation.load_models import load_models

@st.cache_resource
def get_models():
    return load_models()

lr_model, kc_model = get_models()

# ─── Plot Style Helpers ────────────────────────────────────────────────────────
PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

def apply_clean_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#f8fafc")
    ax.figure.patch.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1e293b", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color="#64748b")
    ax.set_ylabel(ylabel, fontsize=9, color="#64748b")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Leaderboard",
    "📈 Aspect Performance",
    "⚡ Efficiency",
    "🧩 Model Comparison",
    "🔎 Review Analyzer",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Leaderboard
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model Leaderboard")
    st.markdown("""
    <div class="desc-box">
      <strong>What this shows:</strong> Overall performance of each model measured by
      <em>Mention Macro F1</em> (did the model correctly detect which aspects were discussed?)
      and <em>Sentiment Macro F1</em> (did it correctly classify the sentiment as positive or negative?).
      Higher is better. A model can be strong at detection but weak at polarity classification — this chart
      reveals both dimensions at once.
    </div>
    """, unsafe_allow_html=True)

    # Summary metric cards
    best_mention = combined.loc[combined["mention_macro_f1"].idxmax()]
    best_sentiment = combined.loc[combined["sentiment_macro_f1"].idxmax()]
    fastest = combined.loc[combined["evaluation_time_sec"].idxmin()]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("🏆 Best Mention F1",    f'{best_mention["mention_macro_f1"]:.3f}',   best_mention["model"])
    col_b.metric("🏆 Best Sentiment F1",  f'{best_sentiment["sentiment_macro_f1"]:.3f}', best_sentiment["model"])
    col_c.metric("⚡ Fastest Inference",  f'{fastest["evaluation_time_sec"]:.2f}s',     fastest["model"])
    col_d.metric("🔢 Models Compared",    len(combined))

    st.divider()

    # Find worst-performing aspect per model
    per_aspect["weakest_aspect"] = per_aspect[["FOOD", "PRICE", "SERVICE", "AMBIENCE"]].idxmin(axis=1)
    per_aspect["weakest_score"]  = per_aspect[["FOOD", "PRICE", "SERVICE", "AMBIENCE"]].min(axis=1)

    # Merge into combined
    display_df = combined.merge(
        per_aspect[["model", "macro_f1", "weakest_aspect", "weakest_score"]],
        on="model", how="left"
    )

    # Rename columns for display
    display_df = display_df.rename(columns={
        "model":               "Model",
        "mention_macro_f1":    "Mention F1",
        "sentiment_macro_f1":  "Sentiment F1",
        "macro_f1":            "Average Macro F1",
        "weakest_aspect":      "Weakest Aspect",
        "weakest_score":       "Weakest F1",
        "misclassified_aspects": "Misclassified Aspect"
    })

    # Select and order columns
    display_df = display_df[[
        "Model", "Mention F1", "Sentiment F1",
        "Average Macro F1", "Weakest Aspect", "Weakest F1", "Misclassified Aspect"
    ]]

    st.dataframe(
        display_df.style
            .format({
                "Mention F1":        "{:.2f}",
                "Sentiment F1":      "{:.2f}",
                "Average Macro F1":   "{:.2f}",
                "Weakest F1":        "{:.2f}",
            })
            .background_gradient(subset=["Mention F1", "Sentiment F1", "Average Macro F1"], cmap="Blues")
            .background_gradient(subset=["Weakest F1"], cmap="Reds")
            .highlight_max(subset=["Mention F1", "Sentiment F1", "Average Macro F1"], color="#bbf7d0")
            .highlight_min(subset=["Weakest F1"], color="#fee2e2"),
        width="stretch" )

    st.divider()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(combined))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], combined["mention_macro_f1"],   width, color=PALETTE[0], label="Mention F1",   alpha=0.88)
    bars2 = ax.bar([i + width/2 for i in x], combined["sentiment_macro_f1"], width, color=PALETTE[1], label="Sentiment F1", alpha=0.88)
    ax.set_xticks(list(x))
    ax.set_xticklabels(combined["model"], fontsize=9)
    for bar in list(bars1) + list(bars2):
        ax.annotate(f"{bar.get_height():.2f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8, color="#334155")
    ax.legend(fontsize=9, framealpha=0.6)
    apply_clean_style(ax, title="Mention vs Sentiment F1 by Model", ylabel="F1 Score")
    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Aspect Performance Heatmap
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Per-Aspect Sentiment F1")
    st.markdown("""
    <div class="desc-box">
      <strong>What this shows:</strong> Each cell represents how well a model classifies sentiment
      for a specific aspect (Food, Price, Service, Ambience).
      Darker blue = higher F1 = better performance on that aspect.
      This heatmap helps identify where each model excels or struggles —
      for example, a model might handle <em>Food</em> well but underperform on <em>Ambience</em>
      due to fewer training examples for that category.
    </div>
    """, unsafe_allow_html=True)

    heatmap_data = per_aspect.set_index("model")[["FOOD", "PRICE", "SERVICE", "AMBIENCE"]]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.6,
            linecolor="#e2e8f0",
            annot_kws={"size": 10},
            ax=ax,
        )
        ax.set_title("Sentiment F1 Heatmap by Aspect", fontsize=11, fontweight="bold", color="#1e293b", pad=10)
        ax.tick_params(axis="x", labelsize=9, colors="#334155")
        ax.tick_params(axis="y", labelsize=9, colors="#334155", rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**Aspect Legend**")
        st.markdown("""
        <span class="aspect-pill pill-food">🍱 FOOD</span><br>
        Ratings about taste, freshness, portion size, or presentation of dishes.<br><br>
        <span class="aspect-pill pill-price">💰 PRICE</span><br>
        Comments on value for money, pricing, or affordability.<br><br>
        <span class="aspect-pill pill-service">🤝 SERVICE</span><br>
        Feedback on staff attitude, speed, or attentiveness.<br><br>
        <span class="aspect-pill pill-ambience">🏮 AMBIENCE</span><br>
        Impressions of the restaurant's atmosphere, décor, or noise level.
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Efficiency
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Speed vs. Performance Trade-off")
    st.markdown("""
    <div class="desc-box">
      <strong>What this shows:</strong> Each point is a model plotted by its
      <em>inference speed</em> (x-axis, in seconds for the test set) against its
      <em>Sentiment Macro F1</em> (y-axis).
      The ideal model sits in the <strong>top-left corner</strong> — fast <em>and</em> accurate.
      Points in the bottom-right are slow <em>and</em> inaccurate.
      Use this view when deciding which model to deploy under latency or cost constraints.
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = PALETTE[:len(combined)]
    for i, (_, row) in enumerate(combined.iterrows()):
        ax.scatter(row["evaluation_time_sec"], row["sentiment_macro_f1"],
                   s=200, color=colors[i % len(colors)], zorder=3, edgecolors="white", linewidth=1.5)
        ax.annotate(
            row["model"],
            (row["evaluation_time_sec"], row["sentiment_macro_f1"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9, color="#334155",
        )
    ax.axhline(combined["sentiment_macro_f1"].mean(), color="#94a3b8", linestyle="--", linewidth=1, label="Avg F1")
    ax.legend(fontsize=9, framealpha=0.5)
    apply_clean_style(ax, title="Inference Speed vs Sentiment F1",
                      xlabel="Inference Time (seconds)", ylabel="Sentiment Macro F1")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div class="desc-box" style="border-left-color:#10b981; margin-top:1rem;">
      <strong>Tip:</strong> For production use in a web app with &lt;1s latency requirements,
      prioritise models to the left of the chart. LLMs (like Qwen via Ollama) tend to be
      slower but may handle rare or complex phrasing better.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Model Comparison (Parameters, Size, Speed, Performance)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧩 Full Model Comparison")
    st.markdown("""
    <div class="desc-box">
      <strong>What this shows:</strong> A unified side-by-side comparison of all three models
      across every dimension — architecture, parameter count, model size on disk, training cost,
      inference speed, and predictive performance. Use this table to understand the full
      cost-benefit profile of each approach before choosing one for deployment.
    </div>
    """, unsafe_allow_html=True)

    # ── Static model metadata ──────────────────────────────────────────────────
    MODEL_META = pd.DataFrame([
        {
            "Model":            "TF-IDF + LR",
            "Architecture":     "Bag-of-Words + Linear",
            "Backbone":         "scikit-learn",
            "Parameters":       "192K",
            "Param Count (M)":  0.2,
            "Size on Disk":     "2.36 MB",
            "Size (MB)":        2.36,
            "Fine-tuned?":      "✅ Yes (task-specific)",
            "Requires GPU?":    "❌ No",
            "Pretrained":       "❌ No",
        },
        {
            "Model":            "KcELECTRA",
            "Architecture":     "Transformer (ELECTRA)",
            "Backbone":         "beomi/KcELECTRA-base-v2022",
            "Parameters":       "127M",
            "Param Count (M)":  127,
            "Size on Disk":     "486 MB",
            "Size (MB)":        485.28,
            "Fine-tuned?":      "✅ Yes (ABSA head)",
            "Requires GPU?":    "Recommended",
            "Pretrained":        "Korean corpus",
        },
        {
            "Model":            "Qwen LLM",
            "Architecture":     "Causal LLM (Transformer)",
            "Backbone":         "Qwen via Ollama",
            "Parameters":       "7B",
            "Param Count (M)":  7000,
            "Size on Disk":     "4.7 GB",
            "Size (MB)":        4700,
            "Fine-tuned?":      "❌ No (prompt only)",
            "Requires GPU?":    "✅ Yes (recommended)",
            "Pretrained":         "Multilingual corpus",
        },
    ])

    # Pull runtime metrics from combined CSV and merge
    runtime = combined[["model", "mention_macro_f1", "sentiment_macro_f1",
                         "evaluation_time_sec", "training_time_sec"]].copy()
    runtime.columns = ["Model_key", "Mention F1", "Sentiment F1",
                        "Inference Time (s)", "Training Time (s)"]

    # Align model name keys  (MODEL_META uses short names; map to CSV names)
    name_map = {
        "TF-IDF + LR":  combined["model"].iloc[0],
        "KcELECTRA":    combined["model"].iloc[1],
        "Qwen LLM":     combined["model"].iloc[2],
    }
    MODEL_META["Model_key"] = MODEL_META["Model"].map(name_map)
    full = MODEL_META.merge(runtime, on="Model_key", how="left").drop(
        columns=["Model_key", "Param Count (M)", "Size (MB)"]
    )

    # ── Master comparison table ────────────────────────────────────────────────
    st.markdown("#### 📋 Master Comparison Table")
    display_cols = [
        "Model", "Architecture", "Backbone", "Parameters", "Size on Disk",
        "Fine-tuned?", "Requires GPU?", "Pretrained",
        "Training Time (s)", "Inference Time (s)",
    ]
    st.dataframe(
        full[display_cols].style
            .format({"Mention F1": "{:.3f}", "Sentiment F1": "{:.3f}",
                     "Inference Time (s)": "{:.2f}", "Training Time (s)": "{:.1f}"})
            .background_gradient(subset=["Inference Time (s)"], cmap="Reds_r")
            .background_gradient(subset=["Training Time (s)"], cmap="Oranges_r"),
        width="stretch",
    )

    st.divider()

    # ── Four-panel chart ───────────────────────────────────────────────────────
    st.markdown("#### 📊 Visual Breakdown")

    models      = MODEL_META["Model"].tolist()
    param_vals  = MODEL_META["Param Count (M)"].tolist()
    size_vals   = MODEL_META["Size (MB)"].tolist()
    train_vals  = combined["training_time_sec"].tolist()
    infer_vals  = combined["evaluation_time_sec"].tolist()
    mention_f1  = combined["mention_macro_f1"].tolist()
    sent_f1     = combined["sentiment_macro_f1"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.55, wspace=0.38)

    def _bar(ax, values, title, ylabel, color, log=False):
        bars = ax.bar(models, values, color=color, alpha=0.88, width=0.45)
        for b in bars:
            h = b.get_height()
            label = f"{h/1000:.1f}GB" if h >= 1000 else (f"{h:.0f}M" if h >= 1 else f"{h:.2f}M")
            ax.annotate(label, (b.get_x() + b.get_width()/2, h),
                        ha="center", va="bottom", fontsize=8, color="#334155")
        if log:
            ax.set_yscale("log")
        apply_clean_style(ax, title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelsize=8)

    def _bar_raw(ax, values, title, ylabel, color, fmt="{:.1f}"):
        bars = ax.bar(models, values, color=color, alpha=0.88, width=0.45)
        for b in bars:
            ax.annotate(fmt.format(b.get_height()),
                        (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color="#334155")
        apply_clean_style(ax, title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelsize=8)

    # Row 1
    _bar(axes[0][0], param_vals, "Parameters (M)", "Parameters (M)",  "#6366f1", log=True)
    _bar(axes[0][1], size_vals,  "Model Size (MB)", "Size (MB)",       "#f59e0b", log=True)
    _bar_raw(axes[0][2], train_vals, "Training Time (s)", "Seconds",   "#ef4444", fmt="{:.0f}s")

    # Row 2
    _bar_raw(axes[1][0], infer_vals, "Inference Time (s)", "Seconds",  "#0ea5e9", fmt="{:.2f}s")
    _bar_raw(axes[1][1], mention_f1, "Mention Macro F1",   "F1",       "#10b981", fmt="{:.3f}")
    _bar_raw(axes[1][2], sent_f1,    "Sentiment Macro F1", "F1",       "#3b82f6", fmt="{:.3f}")

    st.pyplot(fig)

    st.divider()

    # ── Radar / spider chart ───────────────────────────────────────────────────
    st.markdown("#### 🕸️ Model Profile Radar")
    st.markdown("""
    <div class="desc-box" style="border-left-color:#8b5cf6;">
      Each axis is <strong>normalised 0→1</strong> (higher = better).
      <em>Speed</em> axes are inverted so faster models score higher.
      This gives an at-a-glance view of where each model is strong and where it makes trade-offs.
    </div>
    """, unsafe_allow_html=True)

    import numpy as np

    def normalise(vals, invert=False):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        n = [(v - mn) / (mx - mn) for v in vals]
        return [1 - v for v in n] if invert else n

    labels   = ["Mention F1", "Sentiment F1", "Inf. Speed", "Train Speed",
                "Compactness", "Korean\nPretrain"]
    korean   = [0, 1, 1]   # 0=no, 1=yes  (LR, KcELECTRA, Qwen)

    data_norm = list(zip(
        normalise(mention_f1),
        normalise(sent_f1),
        normalise(infer_vals,  invert=True),
        normalise(train_vals,  invert=True),
        normalise(size_vals,   invert=True),
        korean,
    ))

    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    radar_colors = ["#6366f1", "#10b981", "#f59e0b"]
    fig2, ax2 = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    fig2.patch.set_facecolor("white")

    for i, (model_name, row) in enumerate(zip(models, data_norm)):
        vals = list(row) + [row[0]]
        ax2.plot(angles, vals, color=radar_colors[i], linewidth=2, label=model_name)
        ax2.fill(angles, vals, color=radar_colors[i], alpha=0.12)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, fontsize=9, color="#334155")
    ax2.set_yticklabels([])
    ax2.set_ylim(0, 1)
    ax2.grid(color="#e2e8f0", linewidth=0.8)
    ax2.spines["polar"].set_color("#cbd5e1")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, framealpha=0.7)
    ax2.set_title("Normalised Model Profiles", fontsize=11, fontweight="bold",
                  color="#1e293b", pad=18)
    plt.tight_layout()

    col_r, col_note = st.columns([1.4, 1])
    with col_r:
        st.pyplot(fig2)
    with col_note:
        st.markdown("""
        **How to read this radar:**

        | Axis | Meaning |
        |---|---|
        | Mention F1 | Aspect detection accuracy |
        | Sentiment F1 | Polarity classification accuracy |
        | Inf. Speed | Faster inference → outer ring |
        | Train Speed | Less training time → outer ring |
        | Compactness | Smaller model size → outer ring |
        | Korean Pretrain | Pre-trained on Korean text |

        A model that fills the whole hexagon is the best at everything —
        real models always trade off some axes against others.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Live Review Analyzer
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔎 Live Review Analyzer")
    st.markdown("""
    <div class="desc-box">
      <strong>How to use:</strong> Paste any Korean restaurant review in the box below,
      choose a model, and click <em>Analyze</em>.
      The model will detect which aspects (Food, Price, Service, Ambience) are mentioned
      and predict whether the sentiment is <span style="color:#166534;font-weight:600;">Positive</span>,
      <span style="color:#991b1b;font-weight:600;">Negative</span>, or
      <span style="color:#475569;font-weight:600;">Not Mentioned</span>.
    </div>
    """, unsafe_allow_html=True)

    MODEL_DESCRIPTIONS = {
        "TF-IDF + Logistic Regression": (
            "⚙️ **TF-IDF + Logistic Regression** — Classical NLP baseline. "
            "Converts text into word-frequency vectors and uses a linear classifier. "
            "Extremely fast and interpretable, but may miss nuanced or context-dependent expressions."
        ),
        "KcELECTRA": (
            "🤖 **KcELECTRA** — Fine-tuned Korean ELECTRA transformer (`beomi/KcELECTRA-base-v2022`). "
            "Pre-trained on large Korean corpora and fine-tuned on restaurant review data. "
            "Best balance of accuracy and speed for Korean text."
        ),
        "Qwen LLM (Ollama)": (
            "🧠 **Qwen LLM via Ollama** — Large language model running locally. "
            "Uses prompt engineering to perform ABSA without any task-specific fine-tuning. "
            "Most flexible but significantly slower; best for ambiguous or complex reviews."
        ),
    }

    LABEL_MAP = {
        0: ("sent-none", "⚪ Not Mentioned"),
        1: ("sent-neg",  "🔴 Negative"),
        2: ("sent-pos",  "🟢 Positive"),
    }
    col_left, col_right = st.columns([3, 2])
    SAMPLE_REVIEWS = [
    "음식이 정말 맛있었고 서비스도 친절했어요.",
    "가격이 너무 비싸서 실망했어요.",
    "분위기는 좋았지만 음식이 별로였어요.",
    "서비스가 느리고 직원들이 불친절했어요.",
    "음식도 맛있고 가격도 합리적이에요. 또 올게요!",
    "분위기가 너무 시끄러워서 대화하기 힘들었어요.",
    "음식 양이 적은데 가격은 비싸네요.",
    "친절한 서비스와 맛있는 음식 덕분에 즐거운 시간이었어요.",
     "맛은 있는데 가격이 좀 비싸네요.",
    "분위기 좋고 음식도 맛있지만 기다리는 시간이 길어요."]
    
    with col_left:
        selected = st.selectbox(
            "Choose a sample review or write your own:",
            options=["— select a sample —"] + SAMPLE_REVIEWS,
        )

        review = st.text_area(
            "Enter a Korean restaurant review",
            value=selected if selected != "— select a sample —" else "",
            placeholder="예: 음식은 정말 맛있었는데 서비스가 너무 느렸어요. 가격은 적당한 것 같아요.",
            height=130,
        )

        model_choice = st.selectbox(
            "Select Model",
            list(MODEL_DESCRIPTIONS.keys()),
        )
        st.markdown(MODEL_DESCRIPTIONS[model_choice])
        analyze = st.button("▶ Analyze Review", type="primary", use_container_width=True)
        
    with col_right:
        st.markdown("**Aspect Descriptions**")
        st.markdown("""
        | Aspect | What it covers |
        |---|---|
        | 🍱 FOOD | Taste, freshness, portion, presentation |
        | 💰 PRICE | Value, cost, affordability |
        | 🤝 SERVICE | Staff, speed, attentiveness |
        | 🏮 AMBIENCE | Atmosphere, décor, noise |
        """)

    if analyze and review:
        start = time.perf_counter()

        if model_choice == "TF-IDF + Logistic Regression":
            pred = lr_model.predict([review])

        elif model_choice == "KcELECTRA":
            with st.spinner("Running KcELECTRA inference..."):
                pred = kc_model.predict([review])

        elif model_choice == "Qwen LLM (Ollama)":
            with st.spinner("Running Qwen LLM inference (this may take a moment)..."):
                pred = predict_llm_batch(review)

        elapsed = time.perf_counter() - start
        pred_row = pred.iloc[0]

        st.divider()
        st.markdown(f"<div class='time-chip'>⏱ Inference time: {elapsed:.3f}s</div>", unsafe_allow_html=True)

        INLINE_BADGE = {
            0: ("background:#f1f5f9;color:#475569;",  "⚪ Not Mentioned"),
            1: ("background:#fee2e2;color:#991b1b;",  "🔴 Negative"),
            2: ("background:#dcfce7;color:#166534;",  "🟢 Positive"),
        }

        rows_html = ""
        for aspect, value in pred_row.items():
            badge_style, label_text = INLINE_BADGE.get(value, ("background:#f1f5f9;color:#475569;", "⚪ Unknown"))
            rows_html += (
                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'padding:0.55rem 0;border-bottom:1px solid #f1f5f9;">'
                f'<span style="font-weight:600;color:#1e293b;font-size:0.95rem;">{aspect}</span>'
                f'<span style="{badge_style}padding:0.22rem 0.85rem;border-radius:20px;'
                f'font-size:0.82rem;font-weight:600;">{label_text}</span>'
                '</div>'
            )

        # Header row
        header_html = (
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:0.4rem 0;border-bottom:2px solid #e2e8f0;margin-bottom:0.4rem;">'
            '<span style="font-weight:700;color:#475569;font-size:0.85rem;text-transform:uppercase;">Aspects</span>'
            '<span style="font-weight:700;color:#475569;font-size:0.85rem;text-transform:uppercase;">Predictions</span>'
            '</div>'
        )

        st.markdown(
            '<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;'
            'padding:1.2rem 1.6rem;margin-top:1rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);">'
            
            # Model info
            + '<p style="font-size:0.82rem;color:#64748b;margin:0 0 0.6rem 0;">'
            + f'Model: <strong style="color:#1e293b;">{model_choice}</strong></p>'
            
            # Header
            + header_html
            
            # Rows
            + rows_html +
            
            '</div>',
            unsafe_allow_html=True,
    )

    elif analyze and not review:
        st.warning("Please enter a review before clicking Analyze.")
