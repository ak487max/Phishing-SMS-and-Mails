
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def show_dashboard():
    # ---------- File paths ----------
    BASE_DIR = Path(__file__).resolve().parent
    METRICS_PATH = BASE_DIR / "evaluation_metrics.json"

    # ---------- Page heading ----------
    st.title("📊 PhishGuard Evaluation Dashboard")
    st.caption(
        "Performance analysis of your ML-based SMS spam classifier"
    )

    st.divider()

    # ---------- Load evaluation data ----------
    if not METRICS_PATH.exists():
        st.error(
            "Evaluation data not found. "
            "Please run train_model.py first to generate "
            "evaluation_metrics.json."
        )
        return

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            evaluation = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        st.error(f"Could not read evaluation data: {error}")
        return

    report = evaluation["classification_report"]
    accuracy = evaluation["accuracy"]
    test_size = evaluation["test_size"]
    cm = evaluation["confusion_matrix"]
    labels = evaluation["labels"]

    # ---------- KPI metrics ----------
    st.subheader("🎯 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy:.2%}",
    )

    col2.metric(
        "Spam Precision",
        f"{report['Spam']['precision']:.2%}",
    )

    col3.metric(
        "Spam Recall",
        f"{report['Spam']['recall']:.2%}",
    )

    col4.metric(
        "Spam F1-score",
        f"{report['Spam']['f1-score']:.2%}",
    )

    st.caption(
        f"Evaluated on {test_size:,} held-out test messages."
    )

    st.divider()

    # ---------- Confusion matrix ----------
    st.subheader("🔍 Confusion Matrix")

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(6, 4))

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

        plt.setp(
            ax.get_xticklabels(),
            rotation=15,
            ha="right",
        )

        threshold = max(max(row) for row in cm) / 2

        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(
                    j,
                    i,
                    f"{cm[i][j]:,}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if cm[i][j] > threshold
                        else "black"
                    ),
                    fontsize=13,
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("### How to interpret it")

        st.markdown(
            """
            - **Top-left:** Legitimate messages correctly classified.
            - **Top-right:** Legitimate messages incorrectly flagged.
            - **Bottom-left:** Spam messages incorrectly classified
              as legitimate.
            - **Bottom-right:** Spam messages correctly identified.
            """
        )

        st.info(
            "False negatives are especially important in security "
            "because suspicious messages may go undetected."
        )

    st.divider()

    # ---------- Precision, recall, F1 ----------
    st.subheader("📈 Class-wise Performance")

    classes = ["Legitimate", "Spam"]
    metric_names = ["precision", "recall", "f1-score"]

    chart_data = pd.DataFrame(
        {
            metric: [
                report[label][metric]
                for label in classes
            ]
            for metric in metric_names
        },
        index=classes,
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    chart_data.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("Message Class")
    ax.set_title("Precision, Recall and F1-score")
    ax.legend(title="Metric")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Scores range from 0 to 1. Higher scores are generally better, "
        "but precision and recall involve different trade-offs."
    )

    st.divider()

    # ---------- Detailed classification report ----------
    st.subheader("📋 Detailed Classification Report")

    rows = []

    for label in [
        "Legitimate",
        "Spam",
        "macro avg",
        "weighted avg",
    ]:
        row = report[label]

        rows.append(
            {
                "Class": label,
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1-score": row["f1-score"],
                "Support": int(row["support"]),
            }
        )

    report_df = pd.DataFrame(rows).set_index("Class")

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

    st.divider()

    # ---------- Prediction errors ----------
    st.subheader("⚠️ Prediction Error Analysis")

    # Assumes labels are ordered ["Legitimate", "Spam"]
    false_positives = int(cm[0][1])
    false_negatives = int(cm[1][0])
    true_positives = int(cm[1][1])
    true_negatives = int(cm[0][0])

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "False Positives",
            f"{false_positives:,}",
        )
        st.write(
            "Legitimate messages incorrectly classified as spam. "
            "These can cause unnecessary warnings."
        )

    with col2:
        st.metric(
            "False Negatives",
            f"{false_negatives:,}",
        )
        st.write(
            "Spam messages incorrectly classified as legitimate. "
            "These are missed spam messages."
        )

    with st.expander("View all confusion matrix counts"):
        st.write(f"True negatives: {true_negatives:,}")
        st.write(f"True positives: {true_positives:,}")
        st.write(f"False positives: {false_positives:,}")
        st.write(f"False negatives: {false_negatives:,}")

    st.divider()

    # ---------- Model insights ----------
    st.subheader("💡 Model Insights")

    spam_recall = report["Spam"]["recall"]
    spam_precision = report["Spam"]["precision"]

    if spam_recall < 0.80:
        st.warning(
            "Spam recall is below 80%. Consider improving the "
            "training data or testing alternative models."
        )
    else:
        st.success(
            f"The model identifies {spam_recall:.1%} of spam messages "
            "in the held-out test set."
        )

    if spam_precision < 0.80:
        st.warning(
            "Spam precision is below 80%. Review false positives "
            "and consider tuning the classification threshold."
        )

    st.info(
        "Dataset limitation: this model was trained on the SMS Spam "
        "Collection. These metrics evaluate SMS spam classification, "
        "not phishing detection specifically. A dedicated phishing "
        "message dataset is needed to evaluate phishing detection."
    )
