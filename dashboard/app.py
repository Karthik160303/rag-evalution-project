
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "full_evaluation_results.json"
)


# ==========================================
# LOAD JSON REPORT
# ==========================================

@st.cache_data
def load_report():

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Report not found: {REPORT_PATH}"
        )

    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================
# HEADER
# ==========================================

st.title("📊 RAG Evaluation Dashboard")

st.markdown(
    """
    Monitor and analyze the performance of your
    Retrieval Augmented Generation system.
    """
)

if st.button("🔄 Refresh Evaluation Data"):
    st.cache_data.clear()
    st.rerun()


# ==========================================
# LOAD REPORT
# ==========================================

try:
    report = load_report()

except (FileNotFoundError, json.JSONDecodeError) as error:
    st.error(str(error))
    st.stop()


# ==========================================
# EXTRACT METRICS
# ==========================================

retrieval = report.get(
    "average_retrieval_metrics", {}
)

answer = report.get(
    "average_answer_metrics", {}
)

results = report.get("results", [])

if not retrieval or not answer:
    st.error(
        "Required average metrics are missing from the JSON report."
    )
    st.stop()


# ==========================================
# PROJECT INFORMATION
# ==========================================

st.divider()

st.subheader("Project Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Questions",
    report.get("total_questions", 0)
)

col2.metric(
    "Top K",
    report.get("top_k", 0)
)

col3.metric(
    "Evaluated Results",
    len(results)
)


# ==========================================
# RETRIEVAL METRICS
# ==========================================

st.divider()

st.subheader("🔍 Retrieval Evaluation")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Precision@K",
    f"{retrieval['precision_at_k']:.2%}"
)

col2.metric(
    "Recall@K",
    f"{retrieval['recall_at_k']:.2%}"
)

col3.metric(
    "MRR",
    f"{retrieval['mrr']:.2%}"
)

col4.metric(
    "Hit Rate@K",
    f"{retrieval['hit_rate_at_k']:.2%}"
)


# ==========================================
# ANSWER QUALITY METRICS
# ==========================================

st.divider()

st.subheader("🧠 Answer Quality Evaluation")

col1, col2 = st.columns(2)

col1.metric(
    "Faithfulness",
    f"{answer['faithfulness_score']:.2%}"
)

col2.metric(
    "Answer Relevancy",
    f"{answer['answer_relevancy_score']:.2%}"
)


# ==========================================
# INDIVIDUAL QUESTION RESULTS
# ==========================================

st.divider()

st.subheader("📋 Individual Question Results")

if results:

    rows = []

    for item in results:

        retrieval_metrics = item.get(
            "retrieval_metrics", {}
        )

        answer_metrics = item.get(
            "answer_metrics", {}
        )

        rows.append({
            "Question": item.get("question", ""),
            "Precision@K": retrieval_metrics.get(
                "precision_at_k"
            ),
            "Recall@K": retrieval_metrics.get(
                "recall_at_k"
            ),
            "MRR": retrieval_metrics.get("mrr"),
            "Hit Rate@K": retrieval_metrics.get(
                "hit_rate_at_k"
            ),
            "Faithfulness": answer_metrics.get(
                "faithfulness_score"
            ),
            "Answer Relevancy": answer_metrics.get(
                "answer_relevancy_score"
            )
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("No individual results available.")


# ==========================================
# RETRIEVAL COMPARISON CHART
# ==========================================

st.divider()

st.subheader("📈 Retrieval Performance by Question")

if results:

    retrieval_chart_data = []

    for item in results:

        metrics = item.get("retrieval_metrics", {})

        for metric_name, value in metrics.items():

            retrieval_chart_data.append({
                "Question": item["question"],
                "Metric": metric_name,
                "Score": value
            })

    retrieval_df = pd.DataFrame(
        retrieval_chart_data
    )

    fig = px.bar(
        retrieval_df,
        x="Question",
        y="Score",
        color="Metric",
        barmode="group",
        title="Retrieval Metrics Comparison",
        range_y=[0, 1]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# ANSWER QUALITY CHART
# ==========================================

st.divider()

st.subheader("📊 Answer Quality by Question")

if results:

    answer_chart_data = []

    for item in results:

        metrics = item.get("answer_metrics", {})

        answer_chart_data.append({
            "Question": item["question"],
            "Faithfulness": metrics.get(
                "faithfulness_score"
            ),
            "Answer Relevancy": metrics.get(
                "answer_relevancy_score"
            )
        })

    answer_df = pd.DataFrame(answer_chart_data)

    answer_df = answer_df.melt(
        id_vars=["Question"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        answer_df,
        x="Question",
        y="Score",
        color="Metric",
        barmode="group",
        title="Answer Quality Comparison",
        range_y=[0, 1]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# GENERATED ANSWERS
# ==========================================

st.divider()

st.subheader("💬 Generated Answers")

for index, item in enumerate(results):

    question = item.get("question", f"Question {index + 1}")

    with st.expander(question):

        st.markdown("**Generated Answer**")

        st.write(
            item.get("generated_answer", "Not available")
        )

        st.markdown("**Retrieved Chunk IDs**")

        st.write(
            item.get("retrieved_chunk_ids", [])
        )

        answer_metrics = item.get("answer_metrics", {})

        st.markdown("**Evaluator Explanation**")

        st.info(
            answer_metrics.get(
                "explanation",
                "No explanation available."
            )
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "RAG Evaluation Project | "
    "Streamlit + Pandas + Plotly"
)
