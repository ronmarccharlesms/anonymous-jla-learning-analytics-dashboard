"""
dashboard.py
============
Streamlit front-end for an Anonymous Philippine Senior High School Academic Performance Dashboard.

This file is the sole consumer of analysis_engine.py.  All statistical
computation, feature engineering, and modelling logic lives in the engine;
this file handles only page layout, widget state, caching, and chart
rendering.

Page structure
--------------
General Analysis (sidebar page 1)
    Tab 1 — Overview & Trends
        School-year filter, strand KPI tiles, Subject Performance Extremes,
        Grade Distribution box plot, Grade Density KDE overlay.

    Tab 2 — Subject Deep Dive
        Single-subject skew-normal deep-dive, Head-to-Head comparison.

    Tab 3 — Correlations
        Cohort filters, Correlation Heatmap, Scatter Grid (All / Top 20% /
        Bottom 20%).

    Tab 4 — Predictive Outlook
        Macro RF model validation expander, strand/grade/semester prediction
        controls, risk KPI summary, prediction bar chart, full prediction table.

Student Profile (sidebar page 2)
    Search by name or SIS ID (minimum 3 characters).
    Identity KPI card (8 metrics).
    Academic Trajectory (Growth Curve) + Performance Radar (Spider Chart).
    Subject-Level Peer Comparison (Dumbbell Plot).
    Predictive Grade Outlook (current semester, micro RF model).
    Future Term Forecast (untaken curriculum subjects, same micro RF model).
    Full Academic Transcript (sortable table).

Caching strategy
----------------
@st.cache_data    on get_data()           — serialisable DataFrame; loaded once.
@st.cache_resource on get_macro_models()  — non-serialisable sklearn objects.
@st.cache_resource on get_micro_models()  — non-serialisable sklearn objects.

The @st.cache_resource functions use a leading underscore on the DataFrame
parameter (_df) to prevent Streamlit from attempting to hash it.

Deployment
----------
Restricted to institution-authorised users in compliance with the Philippine
Data Privacy Act of 2012 (Republic Act No. 10173).  A public replication
repository with synthetic data is available at [Repository URL].
"""
# for streamlit dashboard
# dashboard.py

import streamlit as st
import analysis_engine as engine
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# chart_config_global — applied to every st.plotly_chart() call in the file.
# scrollZoom is disabled because accidental scroll-wheel zoom is disorienting
# for non-technical users reviewing charts during board meetings.
# Six mode-bar buttons are removed to reduce UI clutter; autoScale and
# resetScale are retained for users who need to restore the default view.
chart_config_global = {
    'scrollZoom': False,
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d'
    ]
}

st.set_page_config(layout="wide", page_title="Philippine Senior High School Academic Performance Dashboard")

# Canvas LMS gradebook export files — one per semester, ordered chronologically.
# The tenth file (AY 2025-2026, Semester 2) is added when the semester completes.
csv_files = [
    "gb_2021-2022_1.csv", "gb_2021-2022_2.csv",
    "gb_2022-2023_1.csv", "gb_2022-2023_2.csv",
    "gb_2023-2024_1.csv", "gb_2023-2024_2.csv",
    "gb_2024-2025_1.csv", "gb_2024-2025_2.csv",
    "gb_2025-2026_1.csv"
]

@st.cache_data
def get_data():
    """Load and harmonise all CSV files.  Cached as a serialisable DataFrame."""
    return engine.load_and_process_data(csv_files)

@st.cache_resource
def get_macro_models(_df):
    """Train cohort-level RF models once per session.

    @st.cache_resource is required (not @st.cache_data) because sklearn model
    objects are not serialisable.  The leading underscore on _df prevents
    Streamlit from attempting to hash the DataFrame argument.
    """
    return engine.train_macro_model(_df)

@st.cache_resource
def get_micro_models(_df):
    """Train student-subject RF models once per session.  See get_macro_models."""
    return engine.train_micro_model(_df)

# Load data at startup; error stops the app to prevent partial renders.
try:
    with st.spinner("Loading student data..."):
        df = get_data()
    if df.empty:
        st.error("No data loaded. Check files."); st.stop()
except Exception as e:
    st.error(f"Error: {e}"); st.stop()

# Sidebar: page navigation, attribution, and admin cache control.
page = st.sidebar.radio("Navigation", ["General Analysis", "Student Profile"])

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by: <br>**Rxxx Sxxx**", unsafe_allow_html=True)
st.sidebar.caption("Education Technology Coordinator <br>Senior High High School 2025", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Admin Controls")
if st.sidebar.button("Clear Cache & Reload Data"):
    st.cache_data.clear()
    st.rerun()

if page == "General Analysis":
    st.title("Philippine Senior High School Academic Performance Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview & Trends", "Subject Deep Dive", "Correlations", "Predictive Outlook"])

    # ── Tab 1: Overview & Trends ───────────────────────────────────────────
    with tab1:
        # Tooltip on the heading explains the two KPI metrics to users who
        # may not be familiar with the distinction between strand average
        # (all subjects) and passing rate (grades >= 75 only).
        st.markdown("""
            <h2 title="A high-level snapshot of the school's population and academic performance. 
            • Strand Average: Mean grade of all enrolled students within the strand. This includes all subjects regardless of difficulty.
            • Passing Rate: Percentage of grades ≥75 on subjects taken within the strand. Calculated per grade.">
            Performance Overview
            </h2>
        """, unsafe_allow_html=True)

        all_years = sorted(df['school_year'].dropna().unique())
        selected_year_overview = st.selectbox("Select School Year for Overview", all_years, index=len(all_years)-1)

        overview_df = df[df['school_year'] == selected_year_overview]

        metrics = engine.get_overview_metrics(overview_df)

        if metrics:
            c1, c2 = st.columns(2)
            c1.metric("Total Students Enrolled", metrics["Total Students"])

            st.write("### Performance by Strand")
            strand_cols = st.columns(len(metrics["Strands"]))
            for i, (strand, data) in enumerate(metrics["Strands"].items()):
                with strand_cols[i]:
                    st.metric(f"{strand} Average", f"{data['Avg']:.2f}")
                    st.metric(f"{strand} Passing", f"{data['PassRate']:.1f}%")
        else:
            st.warning("No data for selected year.")

        st.divider()

        st.subheader("Subject Performance Extremes")
        st.markdown("A quick look at the subjects with the historically **lowest** and **highest** average grades.")

        col_hardest, col_easiest = st.columns(2)

        fig_hard, fig_easy = engine.plot_subject_extremes_split(df, selected_year_overview)

        if fig_hard is not None and fig_easy is not None:
            fig_hard.update_layout(dragmode=False)
            fig_easy.update_layout(dragmode=False)
            with col_hardest:
                st.plotly_chart(fig_hard, use_container_width=True, config=chart_config_global)
            with col_easiest:
                st.plotly_chart(fig_easy, use_container_width=True, config=chart_config_global)
        else:
            st.warning("Data insufficient to plot subject extremes for the current filters.")

        st.divider()

        c_left, c_right = st.columns(2)

        with c_left:
            st.write("### Grade Distributions Across School Years")
            st.markdown("Shows the median and the outlier grades per strand per school year")
            fig_dist = engine.plot_grade_distribution_interactive(df)
            fig_dist.update_layout(dragmode=False)
            st.plotly_chart(fig_dist, use_container_width=True, config=chart_config_global)

        with c_right:
            st.write("### Grade Density Across School Years")
            st.markdown("Shows how much of the students are distributed per grade.")
            fig_dens = engine.plot_grade_density_interactive(df)
            fig_dens.update_layout(dragmode=False)
            st.plotly_chart(fig_dens, use_container_width=True, config=chart_config_global)

    # ── Tab 2: Subject Deep Dive ───────────────────────────────────────────
    with tab2:
        st.subheader("Subject-Level Analysis")
        all_subjects = sorted(df['course'].dropna().unique())
        selected_subj = st.selectbox("Select Subject", all_subjects)
        
        if selected_subj:
            # Calculate metrics specific to this subject (Aggregated across all years)
            subj_data = df[df['course'] == selected_subj]
            avg_grade = subj_data['numeric_grade'].mean()
            pass_rate = (subj_data['numeric_grade'] >= 75).mean() * 100
            std_dev = subj_data['numeric_grade'].std()
            
            # Display Subject KPIs
            k1, k2, k3 = st.columns(3)
            k1.metric("Global Subject Average", f"{avg_grade:.2f}")
            k2.metric("Global Passing Rate", f"{pass_rate:.1f}%")
            k3.metric("Variability (SD)", f"{std_dev:.2f}")
            st.divider()
            
            fig_deep = engine.plot_subject_deep_dive_interactive(df, selected_subj)
            fig_deep.update_layout(dragmode=False)
            st.plotly_chart(fig_deep, use_container_width=True, config=chart_config_global)

        st.divider()

        st.subheader("Subject Comparison (Head-to-Head)")
        st.markdown("Compare the grade distribution of two subjects to see which was historically more difficult or variable.")

        col1, col2 = st.columns(2)
        with col1:
            subject_a = st.selectbox("Select Subject 1", all_subjects, index=0, key='sub1')
        with col2:
            default_idx_2 = 1 if len(all_subjects) > 1 else 0
            subject_b = st.selectbox("Select Subject 2", all_subjects, index=default_idx_2, key='sub2')

        if subject_a and subject_b:
            fig_compare = engine.plot_subject_comparison_interactive(df, subject_a, subject_b)
            fig_compare.update_layout(dragmode=False)
            st.plotly_chart(fig_compare, use_container_width=True, config=chart_config_global)

    # ── Tab 3: Correlations ───────────────────────────────────────────────
    with tab3:
        st.subheader("Subject Correlations")
        c1, c2, c3, c4 = st.columns(4)

        with c1: sel_year   = st.selectbox("School Year",  sorted(df['school_year'].dropna().unique()), key='cor_year')
        with c2: sel_grade  = st.selectbox("Grade Level",  sorted(df['grade_level'].dropna().unique()), key='cor_grade')
        with c3: sel_strand = st.selectbox("Strand",       sorted(df['strand'].dropna().unique()),      key='cor_strand')
        with c4: view_type  = st.selectbox("View Type", [
            "Correlation Heatmap (Interactive)",
            "Scatter Grid (All)",
            "Scatter Grid (Top 20%)",
            "Scatter Grid (Bottom 20%)"
        ])

        if st.button("Analyze Correlations"):

            if "Correlation Heatmap" in view_type:
                with st.spinner("Generating Interactive Heatmap..."):
                    fig, msg = engine.plot_correlation_heatmap_interactive(df, sel_year, sel_grade, sel_strand)
                    if fig:
                        fig.update_layout(dragmode=False)
                        st.plotly_chart(fig, use_container_width=True, config=chart_config_global)
                    else:
                        st.warning(msg)

            elif "Scatter Grid (All)" in view_type:
                with st.spinner("Generating Interactive Grid..."):
                    figs, msg = engine.plot_pairwise_correlations_interactive(df, sel_year, sel_grade, sel_strand)
                    if figs:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, f in enumerate(figs):
                            status_text.text(f"Rendering page {i+1} of {len(figs)}...")
                            f.update_layout(dragmode=False)
                            st.plotly_chart(f, use_container_width=True, config=chart_config_global)
                            progress_bar.progress((i + 1) / len(figs))
                        
                        progress_bar.empty()
                        status_text.empty()
                    else: 
                        st.warning(msg)

            # 3. SCATTER GRID (Top 20%)
            elif "Top 20%" in view_type:
                with st.spinner("Analyzing Top Performers..."):
                    stats_df, stud_df, raw_df, m = engine.get_subgroup_statistics(df, sel_year, sel_grade, sel_strand, 'top')
                    if stats_df is not None:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Count", m['count']); col2.metric("Avg GPA", f"{m['avg_gpa']:.2f}"); col3.metric("Threshold", f"{m['threshold']:.2f}")
                        with st.expander("View Student Report"): st.dataframe(stud_df, use_container_width=True)
                        
                        top_ids = raw_df['student sis'].unique()
                        figs, msg = engine.plot_pairwise_correlations_interactive(df, sel_year, sel_grade, sel_strand, top_students=top_ids)
                        if figs:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, f in enumerate(figs):
                                status_text.text(f"Rendering page {i+1} of {len(figs)}...")
                                st.plotly_chart(f, use_container_width=True, config=chart_config_global)
                                progress_bar.progress((i + 1) / len(figs))
                            
                            progress_bar.empty()
                            status_text.empty()
                        else: 
                            st.warning(msg)

            elif "Bottom 20%" in view_type:
                with st.spinner("Analyzing At-Risk..."):
                    stats_df, stud_df, raw_df, m = engine.get_subgroup_statistics(df, sel_year, sel_grade, sel_strand, 'bottom')
                    if stats_df is not None:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Count", m['count']); col2.metric("Avg GPA", f"{m['avg_gpa']:.2f}"); col3.metric("Threshold", f"{m['threshold']:.2f}")
                        with st.expander("View Student Report"): st.dataframe(stud_df, use_container_width=True)
                        
                        bot_ids = raw_df['student sis'].unique()
                        figs, msg = engine.plot_pairwise_correlations_interactive(df, sel_year, sel_grade, sel_strand, bottom_students=bot_ids)
                        if figs:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, f in enumerate(figs):
                                status_text.text(f"Rendering page {i+1} of {len(figs)}...")
                                st.plotly_chart(f, use_container_width=True, config=chart_config_global)
                                progress_bar.progress((i + 1) / len(figs))
                            
                            progress_bar.empty()
                            status_text.empty()
                        else: 
                            st.warning(msg)

    # ── Tab 4: Predictive Outlook ─────────────────────────────────────────
    with tab4:
        st.subheader("Predictive Outlook — Next Semester Performance")
        st.markdown(
            "Uses a **Random Forest** ensemble model trained on historical cohort data "
            "to forecast subject-level performance for the upcoming semester. "
            "Predictions are generated per strand and grade level. "
            "Colour coding reflects predicted risk relative to the institutional "
            f"threshold of **{engine.AT_RISK_THRESHOLD}**."
        )

        with st.spinner("Loading predictive models... (first load only)"):
            macro_reg, macro_cls, macro_metrics, macro_fi = get_macro_models(df)

        if macro_reg is None:
            err = macro_metrics.get('error', 'Unknown error')
            st.warning(f"Predictive model could not be trained: {err}. "
                       "Ensure at least two complete school years of data are loaded.")
        else:
            with st.expander(
                "📊 Model Validation (Temporal Split — "
                "Train: all years except latest | Test: latest year)"
            ):
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(
                    "MAE (grade pts)",
                    macro_metrics.get('MAE', 'N/A'),
                    help="Mean Absolute Error: average difference in grade points "
                         "between predicted and actual cohort means on the test set."
                )
                m2.metric(
                    "R²",
                    macro_metrics.get('R2', 'N/A'),
                    help="Proportion of variance in cohort mean grades explained by the model."
                )
                m3.metric(
                    "AUC-ROC (At-Risk)",
                    macro_metrics.get('AUC', 'N/A'),
                    help="At-risk classification performance. "
                         "1.0 = perfect discrimination, 0.5 = random."
                )
                m4.metric("Train Samples", macro_metrics.get('train_n', 'N/A'))
                m5.metric("Test Samples",  macro_metrics.get('test_n',  'N/A'))

                if not macro_fi.empty:
                    fi_fig = engine.plot_feature_importance(
                        macro_fi,
                        title="What Historical Factors Drive Cohort Grade Predictions?"
                    )
                    fi_fig.update_layout(dragmode=False)
                    st.plotly_chart(fi_fig, use_container_width=True,
                                    config=chart_config_global)

            st.divider()

            st.markdown("##### Configure Prediction")
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                pred_strand = st.selectbox(
                    "Strand", sorted(df['strand'].dropna().unique()),
                    key='pred_strand'
                )
            with pc2:
                pred_grade = st.selectbox(
                    "Grade Level",
                    sorted(df['grade_level'].dropna().unique()),
                    key='pred_grade'
                )
            with pc3:
                pred_sem = st.selectbox(
                    "Predict for Semester", ['S1', 'S2'],
                    key='pred_sem'
                )

            all_years = sorted(df['school_year'].dropna().unique())
            latest_yr = all_years[-1] if all_years else '2025-2026'
 
            if st.button("Generate Prediction", key='run_macro_pred'):
                with st.spinner("Generating outlook..."):
                    pred_df = engine.predict_macro_outlook(
                        macro_reg, macro_cls, df,
                        pred_strand, pred_grade, pred_sem, latest_yr
                    )
 
                if pred_df.empty:
                    st.warning(
                        "No prediction data for the selected filters. "
                        "Check that historical data exists for this strand and grade."
                    )
                else:
                    # Summary KPI row
                    risk_counts = pred_df['risk_label'].value_counts()
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("🔴 High Risk Subjects",
                               risk_counts.get('🔴 High Risk', 0))
                    rc2.metric("🟡 Moderate Risk Subjects",
                               risk_counts.get('🟡 Moderate', 0))
                    rc3.metric("🟢 On Track Subjects",
                               risk_counts.get('🟢 On Track', 0))
 
                    # Prediction chart
                    fig_macro_pred = engine.plot_macro_prediction_chart(
                        pred_df, pred_strand, pred_grade
                    )
                    fig_macro_pred.update_layout(dragmode=False)
                    st.plotly_chart(fig_macro_pred, use_container_width=True,
                                    config=chart_config_global)
 
                    # Detailed table
                    with st.expander("View Full Prediction Table"):
                        tbl = pred_df[[
                            'course', 'prior_mean',
                            'predicted_mean', 'risk_probability', 'risk_label'
                        ]].copy()
                        tbl.columns = [
                            'Subject', 'Prior Actual Mean',
                            'Predicted Mean', 'Risk Probability', 'Status'
                        ]
                        tbl['Risk Probability'] = tbl['Risk Probability'].apply(
                            lambda p: f"{p * 100:.1f}%"
                        )
                        st.dataframe(tbl, use_container_width=True, hide_index=True)

elif page == "Student Profile":
    st.title("Individual Student Performance Profile")

    # Deduplicated student list used to resolve SIS ID from selected name.
    student_data_unique = df[['student name', 'student sis']].drop_duplicates().reset_index(drop=True)

    search_query = st.text_input("Search Student by Name or ID:", "")

    selected_student_sis = None

    if len(search_query) >= 3:
        query_lower = search_query.lower()

        filtered_students = student_data_unique[
            (student_data_unique['student name'].str.lower().str.contains(query_lower)) |
            (student_data_unique['student sis'].str.contains(query_lower))
        ].sort_values('student name')

        if not filtered_students.empty:
            st.markdown("##### Select Matching Student:")

            # Label is set to "_" (hidden) because the text_input above serves
            # as the visible label.  The selectbox is needed to handle multiple
            # name matches from the same search string.
            selected_name = st.selectbox(
                "_",
                filtered_students['student name'].tolist(),
                key='result_select',
                index=0
            )

            # Where two students share a name, SIS ID is used as the
            # tiebreaker; .iloc[0] takes the first match.
            selected_student_sis = filtered_students[
                filtered_students['student name'] == selected_name
            ].iloc[0]['student sis']

        else:
            st.warning("No students found matching your query.")

    elif len(search_query) > 0:
        st.info("Please type at least 3 characters to search.")

    if selected_student_sis:
        kpis = engine.get_student_kpis(df, selected_student_sis)
        comparison_df = engine.get_subject_performance_vs_peer(df, selected_student_sis)

        st.divider()

        st.subheader(f"Profile: {kpis.get('Name', 'N/A')}")

        col_id, col_strand, col_section, col_gpa = st.columns(4)
        col_id.metric("ID", kpis.get('ID', 'N/A'))
        col_strand.metric("Most Recent Strand", f"{kpis.get('Strand', 'N/A')} ({kpis.get('Latest School Year', 'N/A')})")
        col_section.metric("Most Recent Section", kpis.get('Section', 'N/A'))
        col_gpa.metric("Cumulative GPA", kpis.get('Cumulative GPA', 'N/A'))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Subjects Taken", kpis.get('Total Subjects Taken', 'N/A'))
        c2.metric("Highest Grade", kpis.get('Highest Grade', 'N/A'))
        c3.metric("Lowest Grade", kpis.get('Lowest Grade', 'N/A'))

        with c4:
            class_standing = engine.calculate_class_standing(df, selected_student_sis)
            st.metric("Class Standing (Latest Cohort)", class_standing)

        st.divider()

        # Growth Curve and Spider Chart displayed side-by-side.
        st.subheader("Performance Visualization")
        col_growth, col_spider = st.columns(2)

        with col_growth:
            st.write("#### Academic Trajectory (Term Average)")
            fig_growth = engine.plot_growth_curve(df, selected_student_sis)
            fig_growth.update_layout(dragmode=False)
            st.plotly_chart(fig_growth, use_container_width=True, config=chart_config_global)

        with col_spider:
            st.write("#### Performance Radar (Individual vs Peers)")
            fig_spider = engine.plot_spider_graph(comparison_df)
            fig_spider.update_layout(dragmode=False)
            st.plotly_chart(fig_spider, use_container_width=True, config=chart_config_global)

        st.divider()

        # Dumbbell Plot: subject-by-subject gap between student and peer average.
        st.subheader("Subject-Level Peer Comparison")
        fig_dumb = engine.plot_subject_comparison_dumbbell(comparison_df)
        fig_dumb.update_layout(dragmode=False)
        st.plotly_chart(fig_dumb, use_container_width=True, config=chart_config_global)

        st.divider()

        # ── Predictive Grade Outlook (current semester) ───────────────────────
        st.subheader("Predictive Grade Outlook")
        st.markdown(
            "Predicts this student's subject-level performance based on their "
            "academic trajectory, cumulative GWA trend, and subject difficulty baselines. "
            "Predictions reflect the student's **most recently enrolled semester**."
        )

        with st.spinner("Running student-level predictions..."):
            micro_reg, micro_cls, micro_metrics, micro_fi = get_micro_models(df)

        if micro_reg is None:
            err = micro_metrics.get('error', 'Unknown error')
            st.warning(f"Student-level predictive model unavailable: {err}.")
        else:
            with st.spinner("Generating student outlook..."):
                pred_student_df = engine.predict_student_outlook(
                    micro_reg, micro_cls, df, selected_student_sis
                )

            if pred_student_df.empty:
                st.info(
                    "Not enough historical transcript data for this student "
                    "to generate reliable predictions."
                )
            else:
                risk_counts_s = pred_student_df['risk_label'].value_counts()
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🔴 High Risk",  risk_counts_s.get('🔴 High Risk', 0))
                sc2.metric("🟡 Moderate",   risk_counts_s.get('🟡 Moderate',  0))
                sc3.metric("🟢 On Track",   risk_counts_s.get('🟢 On Track',  0))

                fig_micro_pred = engine.plot_micro_prediction_chart(
                    pred_student_df, kpis.get('Name', 'Student')
                )
                fig_micro_pred.update_layout(dragmode=False)
                st.plotly_chart(fig_micro_pred, use_container_width=True,
                                config=chart_config_global)

                with st.expander("📊 Model Validation & Feature Importance"):
                    mm1, mm2, mm3 = st.columns(3)
                    mm1.metric(
                        "MAE (grade pts)", micro_metrics.get('MAE', 'N/A'),
                        help="Average prediction error in grade points on the test set."
                    )
                    mm2.metric(
                        "R²", micro_metrics.get('R2', 'N/A'),
                        help="Variance explained by the model."
                    )
                    mm3.metric(
                        "AUC-ROC", micro_metrics.get('AUC', 'N/A'),
                        help="At-risk classification performance."
                    )
                    if not micro_fi.empty:
                        fi_fig_m = engine.plot_feature_importance(
                            micro_fi,
                            title="What Predicts Individual Student Grade Outcomes?"
                        )
                        fi_fig_m.update_layout(dragmode=False)
                        st.plotly_chart(fi_fig_m, use_container_width=True,
                                        config=chart_config_global)

        st.divider()

        # ── Future Term Forecast (untaken curriculum subjects) ────────────────
        st.subheader("Future Term Forecast")
        st.markdown(
            "Identifies subjects in the student's strand curriculum that have not yet "
            "been taken and generates predicted grades using the student's current "
            "cumulative GWA as the performance baseline."
        )

        with st.spinner("Generating future forecast..."):
            future_preds = engine.predict_future_performance(
                micro_reg, micro_cls, df, selected_student_sis
            )

        if not future_preds.empty:
            # plot_micro_prediction_chart suppresses the actual-grade trace
            # when numeric_grade is NaN (future subjects), making the
            # forward-looking nature of this section explicit.
            fig_future = engine.plot_micro_prediction_chart(
                future_preds,
                f"Future Forecast: {kpis.get('Name', 'Student')}"
            )
            fig_future.update_layout(dragmode=False)
            st.plotly_chart(fig_future, use_container_width=True, config=chart_config_global)

            with st.expander("View Forecast Details"):
                st.dataframe(
                    future_preds[['course', 'predicted_grade', 'risk_label']].rename(
                        columns={'course': 'Subject', 'predicted_grade': 'Predicted Grade', 'risk_label': 'Risk Status'}
                    ),
                    use_container_width=True, hide_index=True
                )
        else:
            st.info("No upcoming subjects identified for this student's current track or they have completed all levels.")

        st.divider()

        # ── Full Academic Transcript ──────────────────────────────────────────
        st.subheader("Full Academic Transcript")
        transcript_cols = ['course', 'numeric_grade', 'full_term', 'strand', 'section_name']
        transcript_df = df[df['student sis'] == selected_student_sis][transcript_cols].sort_values('full_term', ascending=False)

        st.dataframe(
            transcript_df.rename(columns={
                'course': 'Subject', 'numeric_grade': 'Final Grade',
                'full_term': 'Academic Term', 'strand': 'Strand', 'section_name': 'Section'
            }),
            use_container_width=True,
            hide_index=True
        )