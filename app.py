import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from adis.pipeline import ADISPipeline
import json
from pathlib import Path

# basic page setup
st.set_page_config(
    page_title="ADIS | Automated Data Intelligence System",
    page_icon="🤖",
    layout="wide",
)

# custom styling for a premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e1e2f, #0f0f1a);
        color: #e0e0e0;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .step-badge {
        background: #4facfe;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: inline-block;
    }
    
    .insight-card {
        background: rgba(15, 15, 26, 0.8);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #4facfe;
    }
    
    /* Brighten text inside expanders and general markdown */
    div[data-testid="stExpander"] p, 
    div[data-testid="stExpander"] li,
    div[data-testid="stExpander"] summary,
    .stMarkdown p, 
    .stMarkdown li {
        color: #f4f6f9 !important;
        font-size: 1.05rem;
    }

    /* Premium bright cyan for code blocks */
    code {
        color: #00f2fe !important;
        background-color: rgba(0, 242, 254, 0.1) !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.95em;
    }
</style>
""", unsafe_allow_html=True)

# header section
st.markdown('<h1 class="main-header">ADIS</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated Data Intelligence System • Clean • Analyze • Model</p>', unsafe_allow_html=True)

# sidebar & data upload
with st.sidebar:
    st.image("https://img.icons8.com/wired/512/FFFFFF/workflow.png", width=100) # Generic AI/Workflow Icon
    st.markdown("### Configuration")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    
    target_col = st.text_input("Target Column Name (optional)", placeholder="e.g. price, species")
    
    if st.button("🚀 Run full Pipeline", help="Execute all steps from cleaning to benchmarking"):
        if uploaded_file is not None:
            with st.spinner("ADIS is processing your data..."):
                # Save temp file
                temp_path = Path("temp_data.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                clean_target = target_col.strip() if target_col else None
                pipeline = ADISPipeline(target_column=clean_target)
                results = pipeline.run(str(temp_path))
                st.session_state["adis_results"] = results
                st.success("Pipeline completed successfully!")
        else:
            st.warning("Please upload a CSV file first.")

# main app logic starts here
if "adis_results" in st.session_state:
    results = st.session_state["adis_results"]
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Ingestion", "🧹 Cleaning", "📊 EDA", "🧬 Engineering", "🤖 Modeling", "🛑 Vulnerabilities"])
    
    # Check production safety first
    critic = results.get("critic", {})
    if not critic.get("is_production_safe", True):
        st.error("🚨 **WARNING:** This pipeline is NOT production safe due to critical structural vulnerabilities detected by the AI Critic. Please see the Vulnerabilities tab.", icon="⚠️")
    
    # --- TAB 1: INGESTION ---
    with tab1:
        st.markdown("### Ingestion Overview")
        meta = results["ingestion"]["metadata"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{meta['rows']:,}")
        col2.metric("Columns", meta['columns'])
        col3.metric("Size", f"{meta['file_size_kb']} KB")
        col4.metric("Memory", f"{meta['memory_usage_kb']} KB")
        
        st.write(results["ingestion"]["explanation"]["what_happened"])
        
        # Column Tipes
        st.markdown("#### Detected Column Types")
        col_types = results["ingestion"]["column_info"]
        type_df = pd.DataFrame.from_dict(col_types, orient='index')
        st.dataframe(type_df[['detected_type', 'pandas_dtype', 'unique_count', 'missing_pct']], use_container_width=True)
    
    # --- TAB 2: CLEANING ---
    with tab2:
        st.markdown("### Data Cleaning Actions")
        expl = results["cleaning"]["explanation"]
        st.info(expl["what_happened"])
        
        st.markdown(f"**Rationale:** {expl['why']}")
        st.markdown(f"**Impact:** {expl['impact']}")
        
        with st.expander("🔍 View detailed change log"):
            st.table(results["cleaning"]["log"])

    # --- TAB 3: EDA ---
    with tab3:
        st.markdown("### Exploratory Data Analysis")
        eda_res = results["eda"]
        
        # Select column to visualize
        num_cols = [c for c, i in results["ingestion"]["column_info"].items() if i["detected_type"] == "numeric"]
        if num_cols:
            selected_col = st.selectbox("Visualize Distribution", num_cols)
            fig = px.histogram(results["cleaning"]["df"], x=selected_col, marginal="box", 
                               title=f"Distribution of {selected_col}", template="plotly_dark",
                               color_discrete_sequence=['#4facfe'])
            st.plotly_chart(fig, use_container_width=True)
            
        # Correlations
        if "correlation_matrix" in eda_res["correlations"]:
            st.markdown("#### Feature Correlation Heatmap")
            corr_df = pd.DataFrame(eda_res["correlations"]["correlation_matrix"])
            fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", 
                                 title="Correlation Matrix", template="plotly_dark",
                                 color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 4: FEATURE ENGINEERING ---
    with tab4:
        st.markdown("### Feature Engineering & Selection")
        fe_expl = results["feature_engineering"]["explanation"]
        
        st.info(fe_expl["what_happened"])
        st.markdown(f"**Why Feature Engineering?** {fe_expl['why']}")
        
        with st.expander("🔍 View Feature Transformation Decisions"):
            for log in results["feature_engineering"]["feature_log"]:
                st.markdown(f"""
                - **Added/Modified:** `{log['feature_name']}` (from `{log['source_column']}`)
                - **Action:** `{log['transformation']}`
                - **Trigger:** {log['rationale']}
                - **Benefit:** _{log['expected_benefit']}_
                ---
                """)
        
        if "feature_selection" in results:
            st.markdown("#### Selection Logic")
            fs_expl = results["feature_selection"]["explanation"]
            st.info(fs_expl["what_happened"])
            st.markdown(f"**Impact:** {fs_expl['impact']}")
            
            with st.expander("🗑️ View Dropped Features & Reasons"):
                for dec in results["feature_selection"]["drop_explanations"]:
                    st.markdown(f"- **Dropped `{dec['column']}`**: {dec['explanation']}")

    # --- TAB 5: MODELING ---
    with tab5:
        if "benchmarking" in results:
            bench_res = results["benchmarking"]
            
            if bench_res.get("status") == "success":
                st.markdown("### Model Benchmarking Results")
                best_model = bench_res["best_model"]
                
                # Show Model Recommendation Reason first
                rec_expl = results["model_recommendation"]["explanation"]
                st.info(rec_expl["what_happened"])
                st.markdown(f"**Why these models?** {rec_expl['why']}")
                
                st.success(f"🏆 **Best Performing Model:** {best_model}")
                
                # Comparison Table
                comp_df = pd.DataFrame(bench_res["explanation"]["model_comparison"])
                st.table(comp_df)
                
                # Metrics Visualization
                metrics_data = []
                for res in bench_res["results"]:
                    m = res["metrics"].copy()
                    m["model"] = res["model_name"]
                    metrics_data.append(m)
                
                metrics_df = pd.DataFrame(metrics_data)
                primary = "roc_auc" if "roc_auc" in metrics_df.columns else "r2_score" if "r2_score" in metrics_df.columns else "accuracy"
                
                fig_bench = px.bar(metrics_df, x="model", y=primary, color="model",
                                  title=f"Model Performance ({primary.replace('_', ' ').upper()})",
                                  template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_bench, use_container_width=True)
                
                # Importance of best model
                for res in bench_res["results"]:
                    if res["model_name"] == best_model and res.get("feature_importance_map"):
                        st.markdown("#### Feature Importance (Top 10)")
                        imp_df = pd.DataFrame.from_dict(res["feature_importance_map"], orient='index', columns=['importance'])
                        imp_df = imp_df.sort_values(by='importance', ascending=False).head(10)
                        fig_imp = px.bar(imp_df, orientation='h', template="plotly_dark", 
                                         color_discrete_sequence=['#00f2fe'])
                        fig_imp.update_layout(showlegend=False)
                        st.plotly_chart(fig_imp, use_container_width=True)
                        break
            else:
                st.warning(f"⚠️ **Benchmarking Skipped:** {bench_res.get('reason', 'Insufficient data for modeling.')}")
                st.info(bench_res["explanation"]["why"])
        else:
            st.info("Modeling results will appear here if a target column is specified.")

    # --- TAB 6: VULNERABILITIES (AI CRITIC) ---
    with tab6:
        st.markdown("### AI Critic Vulnerability Report")
        if "critic" in results:
            critic_data = results["critic"]
            expl = critic_data.get("explanation", {})
            st.info(expl.get("what_happened", ""))
            
            vulnerabilities = critic_data.get("vulnerabilities", [])
            
            if not vulnerabilities:
                st.success("✅ **No cross-signal vulnerabilities found!** The pipeline appears structurally sound.")
                
            for v in vulnerabilities:
                sev = v.get("severity", "info")
                if sev == "critical":
                    color = "🔴"
                    st_call = st.error
                elif sev == "warning":
                    color = "🟡"
                    st_call = st.warning
                else:
                    color = "🟢"
                    st_call = st.success
                    
                with st.container():
                    st_call(f"{color} **{v.get('issue')}** (Confidence: {v.get('confidence', 0.0):.2f})")
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown("**Evidence:**")
                        for e in v.get("evidence", []):
                            st.markdown(f"- `{e}`")
                        st.markdown(f"**Impact:** {v.get('impact')}")
                    
                    with c2:
                        st.markdown(f"**Reasoning:** {v.get('reasoning')}")
                        st.markdown("**Recommended Fixes:**")
                        for f in v.get("fix", []):
                            st.markdown(f"- {f}")
                    st.markdown("---")
        else:
            st.info("AI Critic did not run. Ensure the full pipeline has completed.")

else:
    # Landing Page Visual
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; min-height: 400px; flex-direction: column;">
        <img src="https://img.icons8.com/dotty/512/4FACFE/data-configuration.png" width="150" style="opacity: 0.7; margin-bottom: 2rem;">
        <h2 style="color: #666;">Upload a dataset to begin the automated intelligence journey</h2>
        <p style="color: #444; max-width: 600px; text-align: center;">ADIS handles the complex data science pipeline for you, providing clear explanations for every decision made along the way.</p>
    </div>
    """, unsafe_allow_html=True)
