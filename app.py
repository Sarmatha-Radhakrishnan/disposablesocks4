"""
End-to-End Master Analytics Dashboard
--------------------------------------
A Streamlit app covering:
  1. Descriptive & Diagnostic Analysis
  2. Anomaly Detection (Isolation Forest)
  3. RFM & Clustering (K-Means + Hierarchical / Dendrogram + 3D view)
  4. Classification (KNN, Decision Tree, Random Forest, Gradient Boosting)
  5. Regression (Linear, Ridge, Lasso)
  6. Association Rule Mining (Apriori / Market-Basket style bundling)
  7. Prescriptive Recommendations (auto-summarized from the tabs above)

Deploy on Streamlit Community Cloud: push this folder to a GitHub repo
containing app.py + requirements.txt, then point share.streamlit.io at it.
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# Page config & light styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Master Analytics Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .metric-card {
            background-color: #0E3A3E;
            padding: 14px 18px;
            border-radius: 10px;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 End-to-End Master Analytics Dashboard")
st.caption(
    "Incorporating Anomaly Detection, RFM, 3D Clustering, Dendrograms, "
    "Predictive ML, and Association Rule Mining."
)

# ----------------------------------------------------------------------------
# Sidebar — data input (persists across all tabs via session_state)
# ----------------------------------------------------------------------------
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (.xlsx or .csv)", type=["xlsx", "xls", "csv"]
)

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

df = st.session_state.df

if df is None:
    st.info(
        "👈 Upload an Excel (.xlsx) or CSV file from the sidebar to get started. "
        "All 7 tabs below will activate once data is loaded."
    )
    st.stop()

st.sidebar.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols = df.columns.tolist()

# a running results store, used by the Prescriptive tab at the end
if "results" not in st.session_state:
    st.session_state.results = {}

tabs = st.tabs(
    [
        "📊 1. Desc & Diag",
        "🚨 2. Anomalies",
        "🧩 3. RFM & Clusters",
        "🔮 4. Classification",
        "📈 5. Regression",
        "🔗 6. Association Rule Mining",
        "💡 7. Prescriptive",
    ]
)

# ============================================================================
# TAB 1 — DESCRIPTIVE & DIAGNOSTIC
# ============================================================================
with tabs[0]:
    st.header("Descriptive & Diagnostic Analysis")

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing.rename("Missing Count"), use_container_width=True)

    if len(numeric_cols) >= 2:
        st.subheader("Distribution Explorer")
        dist_col = st.selectbox("Choose a numeric column", numeric_cols, key="dist_col")
        fig = px.histogram(df, x=dist_col, nbins=30, marginal="box",
                            title=f"Distribution of {dist_col}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Diagnostic: Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="Teal",
            title="Correlation Matrix (numeric fields)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Diagnostic: Pairwise Relationship")
        c1, c2 = st.columns(2)
        x_var = c1.selectbox("X variable", numeric_cols, index=0, key="diag_x")
        y_var = c2.selectbox("Y variable", numeric_cols, index=min(1, len(numeric_cols) - 1), key="diag_y")
        fig_scatter = px.scatter(df, x=x_var, y=y_var,
                                  title=f"{x_var} vs. {y_var}")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Need at least 2 numeric columns for correlation/scatter views.")

    st.session_state.results["n_rows"] = df.shape[0]

# ============================================================================
# TAB 2 — ANOMALY DETECTION
# ============================================================================
with tabs[1]:
    st.header("Anomaly Detection (Rare & Risky Events)")
    st.write("Defining constraints and isolating outliers using Isolation Forest.")

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns to run anomaly detection.")
    else:
        anomaly_features = st.multiselect(
            "Select features for anomaly detection",
            numeric_cols,
            default=numeric_cols[: min(4, len(numeric_cols))],
        )
        contamination = st.slider("Select Anomaly Contamination Rate (%)", 1, 25, 5) / 100.0

        if len(anomaly_features) >= 2:
            X = df[anomaly_features].dropna()
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(X)
            df_anom = df.loc[X.index].copy()
            df_anom["Anomaly"] = np.where(preds == -1, "Anomaly", "Normal")

            n_anom = int((preds == -1).sum())
            st.success(
                f"Detected {n_anom} anomalies out of {len(X)} rows "
                f"({n_anom / len(X):.1%}) at a {contamination:.0%} contamination rate."
            )

            c1, c2 = st.columns(2)
            xcol = c1.selectbox("X axis", anomaly_features, index=0, key="anom_x")
            ycol = c2.selectbox("Y axis", anomaly_features, index=min(1, len(anomaly_features) - 1), key="anom_y")
            fig = px.scatter(
                df_anom, x=xcol, y=ycol, color="Anomaly",
                color_discrete_map={"Anomaly": "#B85042", "Normal": "#028090"},
                title="Anomaly Detection Scatter",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig_donut = go.Figure(
                data=[go.Pie(
                    labels=["Anomalies", "Typical"],
                    values=[n_anom, len(X) - n_anom],
                    hole=0.55,
                    marker_colors=["#B85042", "#028090"],
                )]
            )
            fig_donut.update_layout(title="Anomaly Rate", showlegend=True)
            st.plotly_chart(fig_donut, use_container_width=True)

            st.session_state.results["n_anomalies"] = n_anom
            st.session_state.results["anomaly_rate"] = n_anom / len(X)
            st.session_state.results["anomaly_sample"] = len(X)
        else:
            st.info("Select at least 2 features above.")

# ============================================================================
# TAB 3 — RFM & CLUSTERING
# ============================================================================
with tabs[2]:
    st.header("RFM Analysis & Advanced Clustering")

    st.subheader("1. RFM Synthesis")
    st.write("Map columns from your data to Recency / Frequency / Monetary, or skip if not applicable.")
    c1, c2, c3 = st.columns(3)
    rec_col = c1.selectbox("Recency column", ["(none)"] + all_cols, key="rfm_r")
    freq_col = c2.selectbox("Frequency column", ["(none)"] + all_cols, key="rfm_f")
    mon_col = c3.selectbox("Monetary column", ["(none)"] + all_cols, key="rfm_m")

    rfm_cols = [c for c in [rec_col, freq_col, mon_col] if c != "(none)"]
    if rfm_cols:
        st.dataframe(df[rfm_cols].head(10), use_container_width=True)
    else:
        st.info("Pick at least one RFM-style column to preview it here.")

    st.subheader("2. Variable Selection & K-Means Clustering")
    cluster_vars = st.multiselect(
        "Select 3-5 variables for clustering",
        numeric_cols,
        default=numeric_cols[: min(4, len(numeric_cols))],
    )

    if len(cluster_vars) >= 2:
        X = df[cluster_vars].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k_range = range(2, min(8, len(X)) if len(X) > 8 else 2)
        inertias, sils = [], []
        from sklearn.metrics import silhouette_score

        valid_ks = []
        for k in range(2, min(6, len(X) - 1) + 1):
            if k >= len(X):
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X_scaled, labels))
            valid_ks.append(k)

        c1, c2 = st.columns(2)
        with c1:
            fig_elbow = px.line(x=valid_ks, y=inertias, markers=True,
                                 labels={"x": "k", "y": "Inertia"}, title="Elbow Method")
            st.plotly_chart(fig_elbow, use_container_width=True)
        with c2:
            fig_sil = px.line(x=valid_ks, y=sils, markers=True,
                               labels={"x": "k", "y": "Score"}, title="Silhouette Scores")
            st.plotly_chart(fig_sil, use_container_width=True)

        k_opt = st.slider("Select Optimal Clusters", min(valid_ks), max(valid_ks),
                           value=min(3, max(valid_ks)))
        km_final = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
        cluster_labels = km_final.fit_predict(X_scaled)
        df_clustered = df.loc[X.index].copy()
        df_clustered["Cluster"] = cluster_labels.astype(str)

        st.subheader("3. 3D Cluster Visualization")
        if len(cluster_vars) >= 3:
            fig3d = px.scatter_3d(
                df_clustered, x=cluster_vars[0], y=cluster_vars[1], z=cluster_vars[2],
                color="Cluster", opacity=0.75, title="3D Cluster View",
            )
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            fig2d = px.scatter(df_clustered, x=cluster_vars[0], y=cluster_vars[1],
                                color="Cluster", title="2D Cluster View")
            st.plotly_chart(fig2d, use_container_width=True)

        st.subheader("4. Hierarchical Clustering (Dendrogram)")
        linkage_method = st.selectbox(
            "Select Linkage Method", ["ward", "complete", "average", "single"]
        )
        sample_for_dendro = X_scaled[: min(150, len(X_scaled))]
        Z = linkage(sample_for_dendro, method=linkage_method)

        import matplotlib.pyplot as plt

        fig_dendro, ax = plt.subplots(figsize=(10, 4))
        dendrogram(Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title(f"Dendrogram ({linkage_method.capitalize()} Linkage)")
        ax.set_ylabel("Distance")
        ax.set_xticks([])
        st.pyplot(fig_dendro)

        st.session_state.results["k_opt"] = k_opt
        st.session_state.results["cluster_vars"] = cluster_vars
        st.session_state.results["cluster_sizes"] = (
            df_clustered["Cluster"].value_counts().to_dict()
        )
    else:
        st.info("Select at least 2 numeric variables to cluster.")

# ============================================================================
# TAB 4 — CLASSIFICATION
# ============================================================================
with tabs[3]:
    st.header("Predictive Classification Models")

    binary_candidates = [c for c in all_cols if df[c].nunique() == 2]
    if not binary_candidates:
        st.warning("No binary (2-value) column found for a classification target. "
                   "Add one to your dataset (e.g. Purchased Y/N) to use this tab.")
    else:
        target_col = st.selectbox("Select target (binary) column", binary_candidates)
        feature_options = [c for c in numeric_cols if c != target_col]
        feature_cols = st.multiselect(
            "Select feature columns", feature_options,
            default=feature_options[: min(5, len(feature_options))],
        )

        if len(feature_cols) >= 1:
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]
            if y.dtype == object or y.dtype.name == "category":
                y = y.astype("category").cat.codes

            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100.0
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            models = {
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }

            rows = []
            roc_data = {}
            cms = {}
            for name, model in models.items():
                model.fit(X_train_s, y_train)
                pred = model.predict(X_test_s)
                proba = (
                    model.predict_proba(X_test_s)[:, 1]
                    if hasattr(model, "predict_proba")
                    else pred
                )
                rows.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, zero_division=0),
                    "Recall": recall_score(y_test, pred, zero_division=0),
                    "F1-Score": f1_score(y_test, pred, zero_division=0),
                })
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_data[name] = (fpr, tpr, auc(fpr, tpr))
                cms[name] = confusion_matrix(y_test, pred)

            results_df = pd.DataFrame(rows)
            st.dataframe(
                results_df.style.background_gradient(cmap="YlGn", subset=["Accuracy", "F1-Score"]),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("ROC Curves")
                fig_roc = go.Figure()
                for name, (fpr, tpr, auc_val) in roc_data.items():
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc_val:.2f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"),
                                              name="Chance", showlegend=True))
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)
            with c2:
                st.subheader("Confusion Matrix")
                cm_model = st.selectbox("Select model for Confusion Matrix", list(models.keys()))
                cm = cms[cm_model]
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                                    labels=dict(x="Predicted", y="Actual"),
                                    title=f"{cm_model} Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)

            perfect = results_df["Accuracy"].min() >= 0.999
            if perfect:
                st.warning(
                    "⚠️ All models are scoring at or near 100% accuracy. This is a strong "
                    "signal of possible **target leakage** (a feature that encodes the "
                    "target) or a trivially separable target — verify your feature set "
                    "before trusting this result for a business decision."
                )

            st.session_state.results["clf_table"] = results_df
            st.session_state.results["clf_perfect_flag"] = bool(perfect)
        else:
            st.info("Select at least 1 feature column.")

# ============================================================================
# TAB 5 — REGRESSION
# ============================================================================
with tabs[4]:
    st.header("Predictive Regression")

    reg_target = st.selectbox("Select target (continuous) column", numeric_cols, key="reg_target")
    reg_feature_options = [c for c in numeric_cols if c != reg_target]
    reg_features = st.multiselect(
        "Select feature columns", reg_feature_options,
        default=reg_feature_options[: min(5, len(reg_feature_options))],
        key="reg_features",
    )

    if len(reg_features) >= 1:
        data = df[reg_features + [reg_target]].dropna()
        X = data[reg_features]
        y = data[reg_target]

        test_size = st.slider("Test set size (%)", 10, 40, 20, key="reg_test_size") / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg_models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
        }
        rows = []
        for name, model in reg_models.items():
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            rows.append({
                "Model": name,
                "R2 Score": r2_score(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            })
        reg_results = pd.DataFrame(rows)
        best_idx = reg_results["R2 Score"].idxmax()
        st.dataframe(
            reg_results.style.highlight_max(subset=["R2 Score"], color="#FFF6C9"),
            use_container_width=True,
        )

        fig_bar = px.bar(reg_results, x="Model", y="R2 Score", color="Model",
                          title="Model Fit (R²)", text_auto=".3f")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.session_state.results["reg_table"] = reg_results
        st.session_state.results["reg_best_model"] = reg_results.loc[best_idx, "Model"]
    else:
        st.info("Select at least 1 feature column.")

# ============================================================================
# TAB 6 — ASSOCIATION RULE MINING
# ============================================================================
with tabs[5]:
    st.header("Association Rule Mining (Bundling Analysis)")
    st.info(
        "Select the item / product columns that indicate purchase (0/1, or Yes/No). "
        "If your data doesn't have basket-style item flags, this tab won't apply."
    )

    binary_like_cols = [c for c in all_cols if df[c].nunique() <= 3]
    basket_cols = st.multiselect("Select item columns for basket analysis", binary_like_cols)

    if len(basket_cols) >= 2:
        basket_df = df[basket_cols].copy()
        for c in basket_cols:
            if basket_df[c].dtype == object:
                basket_df[c] = basket_df[c].astype(str).str.strip().str.lower().isin(
                    ["1", "yes", "y", "true"]
                )
            else:
                basket_df[c] = basket_df[c].astype(bool)

        min_support = st.slider("Minimum support", 0.01, 0.5, 0.05, step=0.01)
        frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            st.warning("No frequent itemsets found at this support threshold. Try lowering it.")
        else:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values("lift", ascending=False)

            if rules.empty:
                st.warning("No association rules found above lift = 1.0.")
            else:
                top_n = st.slider("Show top N rules (by lift)", 3, 20, 4)
                display_rules = rules.head(top_n).copy()
                display_rules["antecedents"] = display_rules["antecedents"].apply(
                    lambda s: ", ".join(list(s))
                )
                display_rules["consequents"] = display_rules["consequents"].apply(
                    lambda s: ", ".join(list(s))
                )
                st.dataframe(
                    display_rules[["antecedents", "consequents", "support", "confidence", "lift"]]
                    .style.format({"support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.4f}"}),
                    use_container_width=True,
                )

                fig_lift = px.bar(
                    display_rules, x=display_rules.index.astype(str), y="lift",
                    title="Association Rule Lift", labels={"x": "Rule"},
                )
                st.plotly_chart(fig_lift, use_container_width=True)

                st.session_state.results["basket_rules"] = display_rules
    else:
        st.info("Select at least 2 item columns above to run association rule mining.")

# ============================================================================
# TAB 7 — PRESCRIPTIVE
# ============================================================================
with tabs[6]:
    st.header("Prescriptive Recommendations")
    st.subheader("Actionable Insights Auto-Summarized From the Tabs Above")

    r = st.session_state.results
    insights = []

    if "cluster_sizes" in r:
        insights.append(
            f"**Targeted Clustering:** K-Means with k={r.get('k_opt')} on "
            f"{', '.join(r.get('cluster_vars', []))} produced clusters sized "
            f"{r['cluster_sizes']}. Use these as distinct marketing segments."
        )
    if "n_anomalies" in r:
        insights.append(
            f"**Anomaly Constraints:** {r['n_anomalies']} of {r.get('anomaly_sample', '?')} rows "
            f"({r.get('anomaly_rate', 0):.1%}) were flagged as outliers. Consider routing this "
            f"high/low-extreme cohort to a dedicated premium or retention offer rather than "
            f"filtering them out."
        )
    if "basket_rules" in r and not r["basket_rules"].empty:
        top_rule = r["basket_rules"].iloc[0]
        insights.append(
            f"**Bundling Strategy:** The strongest rule ({top_rule['antecedents']} → "
            f"{top_rule['consequents']}) has a lift of {top_rule['lift']:.2f} and confidence "
            f"of {top_rule['confidence']:.0%}. Start with this bundle before expanding further."
        )
    if "clf_table" in r:
        best_clf = r["clf_table"].sort_values("F1-Score", ascending=False).iloc[0]
        note = " ⚠️ Validate for leakage before deploying." if r.get("clf_perfect_flag") else ""
        insights.append(
            f"**Model Deployment:** {best_clf['Model']} gave the best F1-Score "
            f"({best_clf['F1-Score']:.3f}) for predicting the classification target.{note}"
        )
    if "reg_table" in r:
        insights.append(
            f"**Spend Forecasting:** {r['reg_best_model']} was the strongest regressor for "
            f"forecasting the selected numeric target — use it as the basis for spend/pricing "
            f"projections."
        )

    if not insights:
        st.info("Run the analyses in the tabs above first — recommendations will populate here automatically.")
    else:
        for i, insight in enumerate(insights, start=1):
            st.markdown(f"{i}. {insight}")
