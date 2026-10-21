
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "evaluation_metrics.json"

st.set_page_config(
    page_title="PhishGuard | Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 PhishGuard Evaluation Dashboard")
st.caption("Performance analysis of the SMS spam classification model")

if not METRICS_PATH.exists():
    st.error(
        "Evaluation data not found. Run train_model.py first "
        "to generate evaluation_metrics.json."
    )
    st.stop()

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    evaluation = json.load(file)

report = evaluation["classification_report"]
accuracy = evaluation["accuracy"]
test_size = evaluation["test_size"]

# ---------- KPI metrics ----------
st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Spam Precision", f"{report['Spam']['precision']:.2%}")
col3.metric("Spam Recall", f"{report['Spam']['recall']:.2%}")
col4.metric("Spam F1-score", f"{report['Spam']['f1-score']:.2%}")

st.caption(f"Evaluation performed on {test_size:,} held-out test messages.")

st.divider()

# ---------- Confusion matrix ----------
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")

    cm = evaluation["confusion_matrix"]
    labels = evaluation["labels"]

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, interpolation="nearest")

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Label",
        ylabel="Actual Label",
        title="Actual vs Predicted",
    )

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # Display the count inside each cell
    threshold = max(max(row) for row in cm) / 2

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(
                j,
                i,
                f"{cm[i][j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i][j] > threshold else "black",
                fontsize=13,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        **How to read it**
        - Top-left: legitimate messages correctly identified.
        - Bottom-right: spam messages correctly identified.
        - Top-right: legitimate messages incorrectly flagged as spam.
        - Bottom-left: spam messages incorrectly classified as legitimate.
        """
    )

with right:
    st.subheader("Precision, Recall & F1-score")

    classes = ["Legitimate", "Spam"]
    metric_names = ["precision", "recall", "f1-score"]

    chart_data = pd.DataFrame(
        {
            metric: [report[label][metric] for label in classes]
            for metric in metric_names
        },
        index=classes,
    )

    st.bar_chart(chart_data, y_label="Score")

    st.caption(
        "Compare performance across both classes. "
        "High accuracy alone can hide weaknesses on the minority class."
    )

st.divider()

# ---------- Classification report ----------
st.subheader("Detailed Classification Report")

table_data = []

for label in ["Legitimate", "Spam", "macro avg", "weighted avg"]:
    row = report[label]

    table_data.append(
        {
            "Class": label,
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1-score": row["f1-score"],
            "Support": int(row["support"]),
        }
    )

report_df = pd.DataFrame(table_data).set_index("Class")

st.dataframe(
    report_df.style.format(
        {
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-score": "{:.2%}",
            "Support": "{:,}",
        }
    ),
    use_container_width=True,
)

# ---------- Interpretation ----------
st.subheader("Model Insights")

spam = report["Spam"]
fn = evaluation["confusion_matrix"][1][0]
fp = evaluation["confusion_matrix"][0][1]

insight1, insight2 = st.columns(2)

with insight1:
    st.metric("False Negatives", f"{fn:,}")
    st.write(
        "Spam messages incorrectly classified as legitimate. "
        "These represent missed spam in this test set."
    )

with insight2:
    st.metric("False Positives", f"{fp:,}")
    st.write(
        "Legitimate messages incorrectly classified as spam. "
        "These represent unnecessary warnings in this test set."
    )

st.info(
    "Scope limitation: this model was trained on the SMS Spam Collection. "
    "These metrics measure SMS spam classification, not phishing detection "
    "performance. A dedicated phishing dataset is needed to evaluate that task."
)
