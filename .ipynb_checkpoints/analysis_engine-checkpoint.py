"""
analysis_engine.py
==================
Backend analytics and modelling engine for the Philippine Senior High School Academic
Performance Dashboard.

This module is intentionally decoupled from the Streamlit front-end so that
every function can be tested, benchmarked, or imported independently of the
UI layer.  dashboard.py is the sole consumer of this module.

Sections
--------
0. Configuration & Mappings
   Canonical subject-name dictionary and shared constants.

1. Data Loading & Cleaning
   CSV ingestion, schema-drift resolution, regex feature extraction,
   and term-ordering for the longitudinal corpus.

2. Statistics
   Cohort overview metrics and subgroup (Top/Bottom 20%) isolation.

3. Interactive Plotting — General Analysis
   Probability density estimation, skew-normal distribution fitting,
   subject-level deep-dive and head-to-head comparison charts.

4. Interactive Correlation Grid
   Pearson correlation heatmap and paginated pairwise scatter grid
   with optional quintile filtering.

5. Student Profile Analysis
   Individual KPI aggregation, class-standing percentile, longitudinal
   growth curve, radial spider chart, and subject-gap dumbbell plot.

6. Predictive Analytics Engine
   Random Forest regression and classification pipelines at both the
   cohort (macro) and student-subject (micro) levels, including a
   curriculum-map-driven future-term forecast.

Dependencies
------------
pandas, numpy, scipy, plotly, scikit-learn, re
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import skewnorm
import re
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

# -----------------------------------------------------------------------------
# 0. CONFIGURATION & MAPPINGS
# -----------------------------------------------------------------------------

# SUBJECT_NAME_MAPPING — deterministic canonical resolution for course names.
#
# Canvas LMS subject names have drifted across nine semesters (AY 2021–2026)
# due to curriculum revisions, strand-specific suffixes, and inconsistent
# administrative encoding.  Each key is a observed variant; each value is the
# single authoritative label used for all downstream grouping and aggregation.
#
# Design rationale: deterministic lookup is preferred over fuzzy-matching
# because the variant population is finite and auditable.  Any new variant
# found in a future export should be added here rather than relaxing the
# matching strategy.
SUBJECT_NAME_MAPPING = {
    '21st Century Literature': '21st Century Literature from the Philippines and the World',
    'Community Engagement, Solidarity and Citizenship': 'Community Engagement, Solidarity and Citizenship',
    'Community Engagement, Solidarity, and Citizenship (GAS)': 'Community Engagement, Solidarity and Citizenship',
    'Community Engagement, Solidarity, and Citizenship (HUMSS)': 'Community Engagement, Solidarity and Citizenship',
    'Creative Non-Fiction': 'Creative Non-Fiction/Sarilaysay',
    'Creative Writing': 'Creative Writing/Malikhaing Pagsulat',
    'Disciplines in Applied Social Sciences': 'Disciplines and Ideas in the Applied Social Sciences',
    'Empowerment Technology': 'Empowerment Technologies',
    'Filipino sa Piling Larangan': 'Filipino sa Piling Larang',
    'General Mathematics (STEM)': 'General Mathematics',
    'General Mathematics (NON-STEM)': 'General Mathematics',
    'Fundamentals of Accountancy, Business and Management 1': 'Fundamentals of Accounting Business and Management 1',
    'Fundamentals of Accountancy, Business, and Management 1': 'Fundamentals of Accounting Business and Management 1',
    'Fundamentals of Accountancy Business and Management 1': 'Fundamentals of Accounting Business and Management 1',
    'Fundamentals of Accounting, Business and Management 1': 'Fundamentals of Accounting Business and Management 1',
    'Fundamentals of Accounting, Business and Marketing 1': 'Fundamentals of Accounting Business and Management 1',
    'Fundamentals of Accountancy, Business and Management 2': 'Fundamentals of Accounting Business and Management 2',
    'Fundamentals of Accountancy, Business, and Management 2': 'Fundamentals of Accounting Business and Management 2',
    'Fundamentals of Accountancy Business and Management 2': 'Fundamentals of Accounting Business and Management 2',
    'Fundamentals of Accounting, Business and Management 2': 'Fundamentals of Accounting Business and Management 2',
    'Fundamentals of Accounting, Business and Marketing 2': 'Fundamentals of Accounting Business and Management 2',
    'Introduction to Philosophy': 'Introduction to the Philosophy of the Human Person',
    'Komunikasyon': 'Komunikasyon at Pananaliksik sa Wika at Kulturang Filipino',
    'Komunikasyon at Pananaliksik': 'Komunikasyon at Pananaliksik sa Wika at Kulturang Filipino',
    'Oral Communication': 'Oral Communication in Context',
    'Oral Communiation in Context': 'Oral Communication in Context',
    'Pagbasa at Pagsusuri ng Ibat Ibang Teksto Tungo sa Pananaliksik': 'Pagbasa at Pagsusuri ng Iba\'t Ibang Teksto Tungo sa Pananaliksik',
    'Physical Education 1': 'Health Optimizing Physical Education 1',
    'Physical Education 2': 'Health Optimizing Physical Education 2',
    'Physical Education 3': 'Health Optimizing Physical Education 3',
    'Physical Education 4': 'Health Optimizing Physical Education 4',
    'Reading and Writing': 'Reading and Writing Skills',
    'Statistics and Probability (STEM)': 'Statistics and Probability',
    'Statistics and Probability (NON-STEM)': 'Statistics and Probability',
    'Trends, Networks and Critical Thinking': 'Trends, Networks and Critical Thinking in the 21st Century Culture',
    'Trends, Networks and Critical Thinking in the 21st Century': 'Trends, Networks and Critical Thinking in the 21st Century Culture',
    'Trends, Networks, and Critical Thinking in the 21st Century': 'Trends, Networks and Critical Thinking in the 21st Century Culture',
    'Trends, Networks and Critical Thinking in the 21st Century Culture': 'Trends, Networks and Critical Thinking in the 21st Century Culture',
    'Trends, Networks, and Critical Thinking in the 21st Century Culture': 'Trends, Networks and Critical Thinking in the 21st Century Culture',
    'Understanding Culture Society and Politics': 'Understanding Culture, Society, and Politics',
    'World Religion': 'Introduction to World Religions and Belief Systems'
}

# -----------------------------------------------------------------------------
# 1. DATA LOADING & CLEANING
# -----------------------------------------------------------------------------

def generate_subject_code(course_name, max_length=15):
    """Return a display-safe truncated label for a subject name.

    Preserves trailing numeric or alphabetic differentiators (e.g., ' 1', ' 2')
    that distinguish sequential subjects such as 'Physical Education 1' from
    'Physical Education 2'.  Used exclusively for radar-chart axis labels where
    long subject names would overlap; full names are always available via hover.

    Parameters
    ----------
    course_name : str
        Canonical subject name (post-SUBJECT_NAME_MAPPING resolution).
    max_length : int, optional
        Maximum character length of the returned label (default 15).

    Returns
    -------
    str
        Truncated label with '...' inserted before any numeric suffix, or
        the original name if it fits within max_length.  Returns 'N/A' for
        non-string inputs.
    """
    if not isinstance(course_name, str):
        return 'N/A'
    
    if len(course_name) <= max_length:
        return course_name

    # Truncate the core name, leaving room for '...' and the suffix
    core_name_truncated = course_name[:max_length-3] 
    
    # Get the last few characters (e.g., ' 1', ' 2')
    unique_suffix = course_name[-3:].strip()
    
    # Check if the suffix contains a number or is part of a common differentiator (like 'S' or 'A')
    if any(c.isdigit() for c in unique_suffix) or len(unique_suffix) > 1:
        # Example: 'Physical Education 1' -> 'Physical Educ...' + ' 1'
        return f"{core_name_truncated.strip()}...{unique_suffix}"
    
    # Otherwise, just use simple truncation
    return f"{core_name_truncated.strip()}..."

def process_section_info(section_code):
    """Parse a Canvas section SIS string and extract strand, grade level, and section name.

    Canvas section SIS identifiers concatenate multiple administrative fields
    in a single string using underscores as delimiters.  Two encoding
    conventions appear in the corpus:

    • Fused format  : '11S07b' — grade and strand initial are merged
      (e.g., '11S' = Grade 11 STEM, '12A' = Grade 12 ABM).
    • Hyphenated format: 'SY_2024_11_S_07b' — grade and strand are separate
      underscore-delimited tokens.

    The function applies the fused-format pattern first via a primary regex
    r'^(11|12)([ASHG])'; if that fails, it falls back to a token scan of the
    underscore-split parts.

    Parameters
    ----------
    section_code : str
        Raw value from the 'section sis' column of a Canvas gradebook CSV.

    Returns
    -------
    tuple of (str or None, str or None, str or None)
        (strand, grade_level, section_name) where strand ∈ {'STEM','ABM',
        'HUMSS','GAS'} and grade_level ∈ {'11','12'}.  Any field that cannot
        be parsed returns None.
    """
    if not isinstance(section_code, str): return None, None, None
    parts = section_code.split('_'); last_part = parts[-1]
    grade = None; strand = None

    # Primary pattern: fused grade+strand initial at the start of the last token.
    # e.g., '11S07b' → grade='11', strand='STEM'
    fused_match = re.search(r'^(11|12)([ASHG])', last_part)
    if fused_match:
        grade = fused_match.group(1); code = fused_match.group(2)
        if code == 'S': strand = 'STEM'
        elif code == 'A': strand = 'ABM'
        elif code == 'H': strand = 'HUMSS'
        elif code == 'G': strand = 'GAS'
        return strand, grade, last_part
        
    if '11' in parts: grade = '11'
    elif '12' in parts: grade = '12'
    
    for p in parts:
        p_upper = p.upper()
        if p_upper in ['11', '12'] or p_upper.startswith('SY'): continue
        if p_upper in ['S', 'STEM']: strand = 'STEM'
        elif p_upper in ['A', 'ABM']: strand = 'ABM'
        elif p_upper in ['H', 'HUMSS']: strand = 'HUMSS'
        elif p_upper in ['G', 'GAS', 'GB']: strand = 'GAS'
            
    return strand, grade, last_part

def clean_subject_names(df):
    """Apply SUBJECT_NAME_MAPPING to the 'course' column of a DataFrame.

    Strips leading/trailing whitespace before matching to prevent missed
    substitutions caused by encoding artefacts in raw CSV exports.  Operates
    in-place on the supplied DataFrame and returns it for chaining.
    """
    if 'course' in df.columns:
        df['course'] = df['course'].str.strip()
        df['course'] = df['course'].replace(SUBJECT_NAME_MAPPING)
    return df

def load_and_process_data(csv_files):
    """Ingest, harmonise, and concatenate all Canvas gradebook CSV files.

    This is the single entry point for raw data.  It resolves all four
    categories of schema-drift documented in the Data Context section of the
    manuscript: semantic inconsistency in course names, section-SIS encoding
    variation, semester-delimiter dropping in terminal records, and structural
    isolation of per-file snapshots.

    Processing steps applied to each file
    --------------------------------------
    1. Load six columns from the Canvas gradebook export CSV.
    2. Standardise course names via SUBJECT_NAME_MAPPING (applied immediately
       so no downstream operation ever sees a non-canonical label).
    3. Extract school year from 'term sis' using two regex patterns to handle
       both 'SY_YYYY' and bare 'YYYY' formats.
    4. Extract semester (S1/S2) with a non-capturing terminator clause
       r'_(S[12])(?:_|$)' that handles terminal records missing a trailing '_'.
    5. Derive strand, grade_level, and section_name via process_section_info().
    6. Construct 'full_term' (e.g., 'G11-S1') for ordered categorical sorting.

    Post-concatenation steps
    ------------------------
    - Cast 'unposted final grade' to float64 via pd.to_numeric with
      errors='coerce' (non-numeric entries such as 'INC' become NaN).
    - Apply a categorical sort order to 'full_term':
      G11-S1 → G11-S2 → G12-S1 → G12-S2.

    Parameters
    ----------
    csv_files : list of str
        Ordered list of Canvas gradebook CSV file paths.  Files that cannot
        be read are skipped with a printed warning; the pipeline continues
        with all successfully loaded files.

    Returns
    -------
    pandas.DataFrame
        Unified corpus with columns: student name, student sis, course,
        section sis, term sis, unposted final grade, raw_year, semester,
        school_year, strand, grade_level, section_name, full_term,
        numeric_grade.  Returns an empty DataFrame if no files load.
    """
    all_data = []
    for file in csv_files:
        try:
            # Load only the six columns consumed by the pipeline.
            df = pd.read_csv(file, usecols=['student name', 'student sis', 'course', 'section sis', 'term sis', 'unposted final grade'])

            # Resolve course-name variants immediately so all downstream
            # operations operate on a schema-stable canonical label set.
            df = clean_subject_names(df)

            # Extract school year; try 'SY_YYYY' first, fall back to bare 'YYYY'.
            df['raw_year'] = df['term sis'].str.extract(r'SY_(\d{4})')
            if df['raw_year'].isna().all(): df['raw_year'] = df['term sis'].str.extract(r'(\d{4})')
            # The (?:_|$) terminator handles terminal-semester records that drop
            # the trailing delimiter in Grade 12 final-semester exports.
            df['semester'] = df['term sis'].str.extract(r'_(S[12])(?:_|$)')[0].fillna('S0')
            df = df.dropna(subset=['raw_year'])
            df['school_year'] = df['raw_year'].apply(lambda x: f"{x}-{int(x)+1}")

            # Extract strand, grade level, and section name from section SIS.
            extracted_info = df['section sis'].apply(process_section_info)
            df['strand'] = [x[0] for x in extracted_info]
            df['grade_level'] = [x[1] for x in extracted_info]
            df['section_name'] = [x[2] for x in extracted_info]

            df['full_term'] = 'G' + df['grade_level'].astype(str) + '-' + df['semester']

            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not process {file}. Error: {e}")
            
    if not all_data: return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    # Non-numeric entries (e.g., 'INC', 'W') become NaN and are excluded
    # from all statistical operations.
    combined_df['numeric_grade'] = pd.to_numeric(combined_df['unposted final grade'], errors='coerce')

    # Apply an ordered categorical to full_term so any sort() or groupby()
    # on this column returns terms in chronological curriculum order.
    term_order = ['G11-S1', 'G11-S2', 'G12-S1', 'G12-S2']
    combined_df['full_term'] = pd.Categorical(combined_df['full_term'], categories=term_order, ordered=True)

    return combined_df

# -----------------------------------------------------------------------------
# 2. STATISTICS
# -----------------------------------------------------------------------------

def get_overview_metrics(df):
    """Compute strand-level summary KPIs for the Overview & Trends tab.

    Parameters
    ----------
    df : pandas.DataFrame
        Corpus filtered to a single school year (passed from dashboard.py).

    Returns
    -------
    dict
        {'Total Students': str, 'Strands': {strand: {'Avg': float, 'PassRate': float}}}
        Pass rate is the percentage of grade records >= 75 (DepEd threshold).
        Returns an empty dict if the DataFrame is empty.
    """
    if df.empty: return {}
    total_students = df['student sis'].nunique()
    strand_stats = df.groupby('strand')['numeric_grade'].agg(['mean', lambda x: (x>=75).mean()*100])
    strand_stats.columns = ['Avg', 'PassRate']
    return {"Total Students": f"{total_students:,}", "Strands": strand_stats.round(2).to_dict('index')}

def get_subgroup_statistics(df, school_year, grade, strand, group_type='top'):
    """Isolate the top or bottom GPA quintile of a cohort and compute subject-level statistics.

    Quintile thresholds are computed from each student's mean grade across all
    subjects taken in the selected school_year × grade × strand cohort.
    Top quintile: students at or above the 80th GPA percentile.
    Bottom quintile: students at or below the 20th GPA percentile.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.
    school_year : str
        e.g. '2024-2025'
    grade : str
        '11' or '12'
    strand : str
        'STEM', 'ABM', 'HUMSS', or 'GAS'
    group_type : {'top', 'bottom'}
        Which quintile to isolate (default 'top').

    Returns
    -------
    tuple of (stats_df, student_summary_df, group_data, metric_dict) or
             (None, None, None, None) if the cohort is empty.

        stats_df          : per-subject count/mean/std/min/max for the subgroup
        student_summary_df: one row per student with their section, GPA, and
                            best/worst subject
        group_data        : raw grade records for the isolated subgroup
        metric_dict       : {'count', 'avg_gpa', 'threshold'}
    """
    subset = df[(df['school_year'] == school_year) & (df['grade_level'] == grade) & (df['strand'] == strand)].copy()
    if len(subset) == 0: return None, None, None, None

    student_gpa = subset.groupby(['student sis', 'student name'])['numeric_grade'].mean()
    if group_type == 'top':
        threshold = student_gpa.quantile(0.8)
        target_students = student_gpa[student_gpa >= threshold].index
    else:
        threshold = student_gpa.quantile(0.2)
        target_students = student_gpa[student_gpa <= threshold].index
        
    target_ids = [x[0] for x in target_students]
    group_data = subset[subset['student sis'].isin(target_ids)].copy()
    
    stats_df = group_data.groupby('course')['numeric_grade'].agg(['count', 'mean', 'std', 'min', 'max']).round(2).sort_values('mean', ascending=False)
    
    def get_student_details(x):
        x = x.sort_values('numeric_grade')
        return pd.Series({
            'Section': x.iloc[0]['section_name'],
            'Average Grade': round(x['numeric_grade'].mean(), 2),
            'Highest Grade': x.iloc[-1]['numeric_grade'],
            'Highest Subject': x.iloc[-1]['course'],
            'Lowest Grade': x.iloc[0]['numeric_grade'],
            'Lowest Subject': x.iloc[0]['course']
        })

    student_summary_df = group_data.groupby(['student name', 'student sis']).apply(get_student_details)
    student_summary_df = student_summary_df.sort_values('Average Grade', ascending=(group_type != 'top'))
    student_summary_df = student_summary_df.reset_index(level='student sis', drop=True)[['Section', 'Average Grade', 'Highest Grade', 'Highest Subject', 'Lowest Grade', 'Lowest Subject']]
    
    metric_dict = {'count': len(target_students), 'avg_gpa': student_gpa[target_students].mean(), 'threshold': threshold}
    return stats_df, student_summary_df, group_data, metric_dict


def plot_subject_extremes_split(df, school_year=None):
    """Render two bar charts showing the five highest- and five lowest-mean subjects per strand.

    Subjects are ranked by their mean numeric grade within the selected school
    year (or across all years when school_year is None).  Colour encoding
    uses a sequential red gradient for low-performing subjects and a sequential
    green gradient for high-performing subjects, with darker shades assigned
    to the most extreme ranks.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.
    school_year : str or None
        If provided, restricts ranking to a single school year.

    Returns
    -------
    tuple of (plotly.graph_objects.Figure or None, plotly.graph_objects.Figure or None)
        (fig_hardest, fig_easiest).  Both are None if the filtered DataFrame
        is empty or contains no valid strand data.
    """
    if school_year:
        clean_df = df[df['school_year'] == school_year].copy()
        title_suffix = f"({school_year})"
    else:
        clean_df = df.copy()
        title_suffix = "(All Time)"
        
    clean_df = clean_df.dropna(subset=['numeric_grade', 'strand'])
    if clean_df.empty: return None, None

    strands = sorted(clean_df['strand'].unique())
    if not strands: return None, None

    ranks = [1, 2, 3, 4, 5]

    # Darker shades assigned to rank 1 (most extreme) within each gradient.
    red_gradient   = ['#4d192b', '#872c4c', '#c13e6c', '#d37898', '#e6b2c4']
    green_gradient = ['#0f5745', '#1b9878', '#26d9ac', '#67e4c5', '#a8f0de']

    def hex_to_rgba(hex_code, alpha=1.0):
        """Convert a hex colour string to an rgba() CSS string."""
        hex_code = hex_code.lstrip('#')
        return f"rgba({int(hex_code[0:2], 16)}, {int(hex_code[2:4], 16)}, {int(hex_code[4:6], 16)}, {alpha})"

    plot_data = []

    for strand in strands:
        strand_data = clean_df[clean_df['strand'] == strand]
        subj_stats = strand_data.groupby('course')['numeric_grade'].mean().reset_index()

        bottom_5 = subj_stats.sort_values('numeric_grade', ascending=True).head(len(ranks)).copy()
        top_5    = subj_stats.sort_values('numeric_grade', ascending=False).head(len(ranks)).copy()

        bottom_5['strand'] = strand; bottom_5['Rank Type'] = 'Hardest'
        top_5['strand']    = strand; top_5['Rank Type']    = 'Easiest'

        bottom_5['Rank_Index']   = range(len(bottom_5))
        top_5['Rank_Index']      = range(len(top_5))
        bottom_5['Display_Rank'] = range(1, len(bottom_5) + 1)
        top_5['Display_Rank']    = range(1, len(top_5) + 1)

        bottom_5['Color_Solid'] = bottom_5['Rank_Index'].apply(lambda x: red_gradient[x] if x < len(red_gradient) else red_gradient[-1])
        top_5['Color_Solid']    = top_5['Rank_Index'].apply(lambda x: green_gradient[x] if x < len(green_gradient) else green_gradient[-1])

        bottom_5['Color_Fill'] = bottom_5['Color_Solid'].apply(lambda x: hex_to_rgba(x, 0.5))
        top_5['Color_Fill']    = top_5['Color_Solid'].apply(lambda x: hex_to_rgba(x, 0.5))

        # X-axis labels encode strand and rank (e.g., 'STEM-1') since the
        # full subject name is available in the hover tooltip.
        bottom_5['X_Label'] = bottom_5.apply(lambda r: f"{r['strand']}-{r['Display_Rank']}", axis=1)
        top_5['X_Label']    = top_5.apply(lambda r: f"{r['strand']}-{r['Display_Rank']}", axis=1)

        bottom_5['Subject_Name'] = bottom_5['course']
        top_5['Subject_Name']    = top_5['course']

        plot_data.extend(bottom_5.to_dict('records'))
        plot_data.extend(top_5.to_dict('records'))

    if not plot_data: return None, None
    plot_df = pd.DataFrame(plot_data)

    order_map = {s: i for i, s in enumerate(strands)}
    plot_df['Strand_Order'] = plot_df['strand'].map(order_map)
    plot_df = plot_df.sort_values(['Strand_Order', 'Rank Type', 'Display_Rank'], ascending=[True, False, True])

    hardest_df = plot_df[plot_df['Rank Type'] == 'Hardest'].copy()
    easiest_df = plot_df[plot_df['Rank Type'] == 'Easiest'].copy()

    fig_hard = go.Figure(data=[
        go.Bar(
            x=hardest_df['X_Label'],
            y=hardest_df['numeric_grade'],
            marker_color=hardest_df['Color_Fill'].tolist(),
            marker_line_color=hardest_df['Color_Solid'].tolist(),
            marker_line_width=2,
            name='Hardest Subjects',
            text=hardest_df['numeric_grade'].apply(lambda x: f"{x:.1f}"),
            textposition='outside',
            customdata=np.column_stack((hardest_df['Subject_Name'], hardest_df['strand'])),
            hovertemplate="<b>%{customdata[0]}</b><br>Strand: %{customdata[1]}<br>Average: %{y:.1f}<extra></extra>"
        )
    ])

    fig_hard.update_layout(
        title=f"🔴 Lowest Mean Grades {title_suffix}",
        height=400,
        yaxis_title="Average Grade",
        yaxis_range=[70, 101],
        showlegend=False,
        xaxis=dict(title="Subjects by Strand"),
        bargap=0.1,
        bargroupgap=0.1
    )

    fig_easy = go.Figure(data=[
        go.Bar(
            x=easiest_df['X_Label'],
            y=easiest_df['numeric_grade'],
            marker_color=easiest_df['Color_Fill'].tolist(),
            marker_line_color=easiest_df['Color_Solid'].tolist(),
            marker_line_width=2,
            name='Easiest Subjects',
            text=easiest_df['numeric_grade'].apply(lambda x: f"{x:.1f}"),
            textposition='outside',
            customdata=np.column_stack((easiest_df['Subject_Name'], easiest_df['strand'])),
            hovertemplate="<b>%{customdata[0]}</b><br>Strand: %{customdata[1]}<br>Average: %{y:.1f}<extra></extra>"
        )
    ])

    fig_easy.update_layout(
        title=f"🟢 Highest Mean Grades {title_suffix}",
        height=400,
        yaxis_title="Average Grade",
        yaxis_range=[70, 101],
        showlegend=False,
        xaxis=dict(title="Subjects by Strand"),
        bargap=0.1,
        bargroupgap=0.1
    )

    return fig_hard, fig_easy
# -----------------------------------------------------------------------------
# 3. INTERACTIVE PLOTTING (General Analysis)
# -----------------------------------------------------------------------------

def get_color_map(df):
    """Return a consistent school-year → hex-colour mapping.

    Colours are drawn from Plotly's 'Bold' qualitative palette and assigned
    in chronological school-year order.  Reusing this mapping across all
    General Analysis charts ensures that a given school year is always
    represented by the same colour regardless of which chart it appears in.
    """
    years = sorted(df['school_year'].dropna().unique())
    colors = px.colors.qualitative.Bold * 2
    return {year: colors[i % len(colors)] for i, year in enumerate(years)}

def plot_grade_distribution_interactive(df):
    """Render a longitudinal box plot of grade distributions by strand and school year.

    Each box encodes the median, interquartile range, and outlier grades for
    one strand × school-year combination.  Faceting by strand allows direct
    visual comparison of distributional spread across the four curriculum tracks.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    color_map = get_color_map(df)
    fig = px.box(
        df, 
        x='strand', 
        y='numeric_grade', 
        color='school_year', 
        labels={'numeric_grade': 'Final Grade', 'strand': 'Strand', 'school_year': 'School Year'}, 
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        xaxis_title="Strand",
        yaxis_title="Grade",
        # Legend placed below to avoid overlap when multiple school years are present.
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.2,
            xanchor="center", x=0.5
        ),
        margin=dict(b=80)
    )
    return fig

def plot_grade_density_interactive(df):
    """Overlay Gaussian KDE density curves for each school year on a single axis.

    Each curve is computed via scipy.stats.gaussian_kde with Silverman's
    rule-of-thumb bandwidth.  The overlaid curves make population-level grade
    mass migration across years legible in a single chart — a pattern that is
    invisible in the strand-average KPI tiles.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.

    Returns
    -------
    plotly.graph_objects.Figure
        Y-axis is density expressed as a percentage of the student population.
    """
    fig = go.Figure()
    color_map = get_color_map(df)
    years = sorted(df['school_year'].dropna().unique())

    for year in years:
        subset = df[df['school_year'] == year]['numeric_grade'].dropna()
        if len(subset) > 1:
            kde = stats.gaussian_kde(subset)
            x_range = np.linspace(subset.min(), subset.max(), 200)
            y_density = kde(x_range)
            y_percent = y_density * 100
            count = len(subset)
            label = f"{year} (N={count})"

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_percent,
                mode='lines',
                name=label,
                fill='tozeroy',
                line=dict(color=color_map.get(year, 'blue'), width=2),
                opacity=0.3,
                hovertemplate="<b>%{fullData.name}</b><br>Grade: %{x:.1f}<br>Density: %{y:.1f}%<extra></extra>"
            ))
            
    fig.update_layout(
        xaxis_title="Numeric Grade",
        yaxis_title="Density (%)",
        # 'closest' mode shows tooltip only for the curve under the cursor,
        # preventing overlap when multiple years are displayed together.
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.2,
            xanchor="center", x=0.5
        ),
        margin=dict(b=80)
    )
    return fig


def plot_subject_deep_dive_interactive(df, subject_name):
    """Render a per-year skew-normal distribution deep-dive for a single subject.

    For each school year that contains at least two grade records, this function
    plots three superimposed elements:
      • A histogram (opacity 0.25) showing the actual grade frequency distribution.
      • A skew-normal probability density curve fitted via scipy.stats.skewnorm.fit(),
        scaled to percentage-of-students units to match the histogram y-axis.
      • A dashed vertical line at the arithmetic mean.

    The legend entry for each year includes μ (mean) and the skewness shape
    parameter α, making year-on-year distributional drift immediately legible
    without requiring numerical comparison between years.

    Design note: a try/except block wraps skewnorm.fit() because fitting can
    fail on degenerate distributions (e.g., all grades identical).  In those
    cases the histogram and mean line are still rendered; only the smooth curve
    is omitted.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.
    subject_name : str
        Canonical subject name (post-SUBJECT_NAME_MAPPING).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    subset = df[df['course'] == subject_name]
    if len(subset) == 0: return go.Figure()

    fig = go.Figure()

    colors = px.colors.qualitative.Bold * 2
    years = sorted(subset['school_year'].dropna().unique())

    all_grades = subset['numeric_grade'].dropna()
    min_grade = all_grades.min() if not all_grades.empty else 75
    max_y_val = 0

    for i, year in enumerate(years):
        data = subset[subset['school_year'] == year]['numeric_grade'].dropna()
        if len(data) > 1:
            mu = data.mean()
            std = data.std()
            skew = stats.skew(data)

            color = colors[i % len(colors)]
            label = f"{year} (μ={mu:.1f}, Skew={skew:.2f})"

            # Histogram — provides the empirical context for the fitted curve.
            counts, _ = np.histogram(data, bins=np.arange(min(data), max(data)+2, 1))
            if len(counts) > 0:
                max_hist_pct = (counts.max() / len(data)) * 100
                max_y_val = max(max_y_val, max_hist_pct)

            # Hover tooltip uses a filled-block glyph to visually match the
            # bar colour without requiring the user to read a legend entry.
            hist_hover_label = (
                f"<span style='color:{color}; font-size:16px'><b>▍</b></span> " +
                f"<b>{year} Actual</b>: %{{y:.1f}}%<extra></extra>"
            )

            fig.add_trace(go.Histogram(
                x=data,
                name=label,
                legendgroup=year,
                histnorm='percent',
                marker_color=color,
                opacity=0.25,
                xbins=dict(size=1.0),
                hovertemplate=hist_hover_label
            ))

            # Skew-normal fit — wrapped in try/except because fitting fails
            # on degenerate distributions (e.g., all grades identical).
            try:
                a, loc, scale = skewnorm.fit(data)

                # Evaluate the fitted PDF over the grade range and scale to
                # percentage-of-students units for y-axis consistency.
                x = np.linspace(max(60, min_grade-5), 100, 200)
                p = skewnorm.pdf(x, a, loc, scale)
                p_scaled = p * 100

                max_y_val = max(max_y_val, p_scaled.max())

                # Wavy-line glyph in the tooltip distinguishes the smooth
                # fitted curve from the empirical histogram bars.
                curve_hover_label = (
                    f"<span style='color:{color}; font-size:16px'><b>∿</b></span> " +
                    f"<b>{year} Fit</b>: Skew {a:.2f}<extra></extra>"
                )

                fig.add_trace(go.Scatter(
                    x=x,
                    y=p_scaled,
                    mode='lines',
                    name=label,
                    legendgroup=year,
                    showlegend=False,
                    line=dict(color=color, width=3),
                    fill='tozeroy',
                    opacity=0.1,
                    hovertemplate=curve_hover_label
                ))
            except Exception:
                pass

            # Dashed mean line — grouped with the histogram/curve so that
            # toggling a year in the legend hides all three elements together.
            fig.add_trace(go.Scatter(
                x=[mu, mu], 
                y=[0, 100], 
                mode='lines',
                legendgroup=year,
                showlegend=False,
                line=dict(color=color, width=2, dash="dash"),
                hoverinfo='skip'
            ))

    dynamic_min = max(60, min_grade - 4)

    fig.update_layout(
        title=f"Deep Dive: {subject_name} (Skew-Normal Fit)",
        xaxis_title="Numeric Grade",
        yaxis_title="Percentage of Students (%)",
        barmode='overlay',
        hovermode="x unified",
        height=600, 
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.2,
            xanchor="center", x=0.5,
            font=dict(size=12)
        ),
        margin=dict(b=100),
        xaxis=dict(range=[dynamic_min, 100.5], dtick=5),
        yaxis=dict(range=[0, max_y_val * 1.15]) 
    )
    return fig

def plot_subject_comparison_interactive(df, subject1, subject2):
    """
    Side-by-side comparison using Skew-Normal Fitting with styled hover.
    """
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(f"{subject1}", f"{subject2}"),
        horizontal_spacing=0.08
    )
    
    subjects = [subject1, subject2]
    colors = px.colors.qualitative.Bold * 2
    
    # Pre-Calculate Global Max Y for Shared Axis
    def get_max_density(sub_name):
        sub_max = 0
        s_data = df[df['course'] == sub_name]
        years = s_data['school_year'].dropna().unique()
        for y in years:
            vals = s_data[s_data['school_year'] == y]['numeric_grade'].dropna()
            if len(vals) > 1:
                try:
                    a, loc, scale = skewnorm.fit(vals)
                    x = np.linspace(60, 100, 200)
                    p = skewnorm.pdf(x, a, loc, scale) * 100 
                    sub_max = max(sub_max, p.max())
                except: pass
        return sub_max

    global_max_y = max(get_max_density(subject1), get_max_density(subject2))
    global_max_y = max(10, global_max_y * 1.15) 

    # Determine Shared X-Axis Range
    min_grade_1 = df[df['course'] == subject1]['numeric_grade'].min()
    min_grade_2 = df[df['course'] == subject2]['numeric_grade'].min()
    if pd.isna(min_grade_1): min_grade_1 = 75
    if pd.isna(min_grade_2): min_grade_2 = 75
    
    global_min_grade = min(min_grade_1, min_grade_2)
    dynamic_min = max(60, global_min_grade - 4)

    legend_tracker = set() 
    
    for idx, sub_name in enumerate(subjects):
        col_idx = idx + 1
        subset = df[df['course'] == sub_name]
        years = sorted(subset['school_year'].dropna().unique())
        
        for i, year in enumerate(years):
            data = subset[subset['school_year'] == year]['numeric_grade'].dropna()
            if len(data) > 1:
                mu = data.mean()
                
                # Match colors by Year
                all_years = sorted(df['school_year'].dropna().unique())
                color_idx = all_years.index(year) if year in all_years else i
                color = colors[color_idx % len(colors)]
                
                label = f"{year}" 
                show_legend = True if year not in legend_tracker else False
                if show_legend: legend_tracker.add(year)

                # A. HISTOGRAM HOVER STYLE
                hist_hover = (
                    f"<span style='color:{color}; font-size:16px'><b>▍</b></span> " + 
                    f"<b>{year} Data</b>: %{{y:.1f}}%<extra></extra>"
                )

                fig.add_trace(go.Histogram(
                    x=data, 
                    name=label, 
                    legendgroup=year, 
                    histnorm='percent', 
                    marker_color=color, 
                    opacity=0.25, 
                    xbins=dict(size=1.0),
                    showlegend=show_legend,
                    hovertemplate=hist_hover
                ), row=1, col=col_idx)
                
                # B. SMOOTH CURVE HOVER STYLE
                try:
                    a, loc, scale = skewnorm.fit(data)
                    x = np.linspace(dynamic_min, 100, 200) 
                    p = skewnorm.pdf(x, a, loc, scale) * 100
                    
                    curve_hover = (
                        f"<span style='color:{color}; font-size:16px'><b>∿</b></span> " + 
                        f"<b>{year} Fit</b>: Skew {a:.2f}<extra></extra>"
                    )

                    fig.add_trace(go.Scatter(
                        x=x, 
                        y=p, 
                        mode='lines', 
                        name=label, 
                        legendgroup=year,    
                        showlegend=False,    
                        line=dict(color=color, width=2.5), 
                        fill='tozeroy',      
                        opacity=0.1,
                        hovertemplate=curve_hover
                    ), row=1, col=col_idx)
                except: pass
                
                # C. MEAN LINE
                fig.add_trace(go.Scatter(
                    x=[mu, mu], 
                    y=[0, 100], 
                    mode='lines',
                    legendgroup=year,
                    showlegend=False,
                    line=dict(color=color, width=1.5, dash="dash"),
                    hoverinfo='skip'
                ), row=1, col=col_idx)

    fig.update_layout(
        title=f"Head-to-Head: {subject1} vs. {subject2}",
        barmode='overlay',
        hovermode="x unified", 
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.2,
            xanchor="center", x=0.5
        ),
        margin=dict(t=60, b=60)
    )
    
    # Shared Axis Settings
    fig.update_xaxes(title_text="Grade", range=[dynamic_min, 100.5], dtick=5, row=1, col=1)
    fig.update_xaxes(title_text="Grade", range=[dynamic_min, 100.5], dtick=5, row=1, col=2)
    
    fig.update_yaxes(title_text="Density (%)", range=[0, global_max_y], row=1, col=1)
    fig.update_yaxes(range=[0, global_max_y], showticklabels=True, row=1, col=2) 
    
    return fig

# -----------------------------------------------------------------------------
# 4. INTERACTIVE CORRELATION GRID
# -----------------------------------------------------------------------------

def truncate_title(text, limit=25):
    """Truncates text to limit and adds '...' if longer."""
    return text if len(text) <= limit else text[:limit] + "..."

def plot_pairwise_correlations_interactive(df, school_year, grade, strand, top_students=None, bottom_students=None):
    """Render a paginated scatter grid of all pairwise subject correlations.

    Each subplot shows individual student grades on two subjects as a scatter
    plot, with an OLS trendline and two background performance zones:
      • Green zone (both subjects ≥ 80): high co-performance quadrant.
      • Red zone  (both subjects ≤ 80): low co-performance quadrant.

    Student identity is encoded in the Plotly customdata hover tooltip so that
    coordinators can immediately identify which student occupies a given point.

    Quintile filtering: if top_students or bottom_students is supplied, the
    function restricts the scatter to those student IDs only, rendering the
    correlation structure of the subgroup in isolation.  This capability is
    used by the Correlations tab to compare Top 20% vs. Bottom 20% GPA cohorts.

    Plots are paginated at max_plots=16 per figure to prevent browser
    performance degradation on strands with many subjects.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.
    school_year, grade, strand : str
        Filters identifying the target cohort.
    top_students : array-like of str, optional
        SIS IDs of the top GPA quintile (from get_subgroup_statistics).
    bottom_students : array-like of str, optional
        SIS IDs of the bottom GPA quintile.

    Returns
    -------
    tuple of (list[plotly.graph_objects.Figure], str)
        (figures, status_message). figures is empty and message is descriptive
        if the cohort has insufficient data.
    """
    subset = df[(df['school_year'] == school_year) & (df['grade_level'] == grade) & (df['strand'] == strand)].copy()
    if len(subset) == 0: return [], "No Data"
    
    title_prefix = f"All Students: {strand} - Grade {grade}"
    if top_students is not None:
        subset = subset[subset['student sis'].isin(top_students)]
        title_prefix = f"Top 20%: {strand} - Grade {grade}"
    elif bottom_students is not None:
        subset = subset[subset['student sis'].isin(bottom_students)]
        title_prefix = f"Bottom 20%: {strand} - Grade {grade}"

    pivot_df = subset.pivot_table(index=['student sis', 'student name'], columns='course', values='numeric_grade')
    pivot_df = pivot_df.dropna(axis=1, thresh=3)
    subjects = pivot_df.columns.tolist()
    if len(subjects) < 2: return [], "Not enough subjects to correlate."

    subject_pairs = [(s1, s2) for i, s1 in enumerate(subjects) for s2 in subjects[i+1:]]
    figures = []
    n_pairs = len(subject_pairs)
    max_plots = 16
    n_figures = (n_pairs + max_plots - 1) // max_plots

    # Dot colour matches the school-year colour from get_color_map so that
    # scatter plots are visually consistent with the Overview & Trends tab.
    years = sorted(df['school_year'].dropna().unique())
    colors = px.colors.qualitative.Bold * 2
    if school_year in years:
        color_idx = years.index(school_year)
        dot_color = colors[color_idx % len(colors)]
    else:
        dot_color = colors[0]

    trend_color = '#FFFFFF'   # white trendline for contrast on dark background
    green_zone = '#81C784'    # co-high-performance zone (both subjects ≥ 80)
    red_zone = '#E57373'      # co-low-performance zone  (both subjects ≤ 80)
    zone_opacity = 0.15       
    
    for fig_num in range(n_figures):
        start = fig_num * max_plots
        end = min((fig_num + 1) * max_plots, n_pairs)
        current_pairs = subject_pairs[start:end]
        
        n_curr = len(current_pairs)
        n_cols = min(4, n_curr)
        n_rows = (n_curr + n_cols - 1) // n_cols
        
        titles = [f"{truncate_title(s1)}<br>vs {truncate_title(s2)}" for s1, s2 in current_pairs]

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=titles,
            horizontal_spacing=0.10, 
            vertical_spacing=0.12    
        )
        
        for idx, (s1, s2) in enumerate(current_pairs):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            pair_data = pivot_df[[s1, s2]].dropna().reset_index()
            
            if len(pair_data) > 0:
                # Scatter
                fig.add_trace(go.Scatter(
                    x=pair_data[s1], y=pair_data[s2], mode='markers',
                    marker=dict(size=5, color=dot_color, line=dict(width=0)),
                    text=pair_data['student name'],
                    customdata=np.stack((
                        pair_data['student name'], 
                        [s1]*len(pair_data), 
                        [s2]*len(pair_data), 
                        pair_data['student sis']
                    ), axis=-1),
                    hovertemplate="<b>%{customdata[0]}</b> (%{customdata[3]})<br>%{customdata[1]}: %{x}<br>%{customdata[2]}: %{y}<extra></extra>"
                ), row=row, col=col)
                
                # Trendline
                if len(pair_data) > 1:
                    z = np.polyfit(pair_data[s1], pair_data[s2], 1)
                    fig.add_trace(go.Scatter(
                        x=np.linspace(60, 100, 10), y=np.poly1d(z)(np.linspace(60, 100, 10)), 
                        mode='lines', line=dict(color=trend_color, width=1.5, dash='dash'), showlegend=False
                    ), row=row, col=col)
                
                # Zones
                fig.add_shape(type="rect", x0=80, y0=80, x1=100, y1=100, fillcolor=green_zone, opacity=zone_opacity, layer="below", line_width=0, row=row, col=col)
                fig.add_shape(type="rect", x0=60, y0=60, x1=80, y1=80, fillcolor=red_zone, opacity=zone_opacity, layer="below", line_width=0, row=row, col=col)
                
                # AXES
                fig.update_xaxes(range=[60, 100], row=row, col=col, dtick=5, showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(range=[60, 100], row=row, col=col, dtick=5, showgrid=True, gridcolor='rgba(255,255,255,0.1)', scaleanchor="x", scaleratio=1)

        fig.update_annotations(font_size=11)

        fig.update_layout(
            height=320*n_rows,
            width=1100, 
            title_text=f"{title_prefix} (Page {fig_num+1}/{n_figures})",
            showlegend=False,
            template='plotly_dark',
            margin=dict(t=80, b=50, l=50, r=50)
        )
        figures.append(fig)
        
    return figures, "Success"

def plot_correlation_heatmap_interactive(df, school_year, grade, strand):
    """Render an interactive Pearson correlation matrix heatmap.

    Computes pairwise Pearson product-moment correlations across all subjects
    for the selected cohort.  Subjects with fewer than five student records
    are excluded (dropna thresh=5) to prevent unreliable correlations from
    being displayed.

    The Purpor diverging colour scale maps +1.0 (maximum positive correlation)
    to deep purple and -1.0 (maximum negative correlation) to near-white,
    making strong positive relationships immediately salient.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.
    school_year, grade, strand : str
        Filters identifying the target cohort.

    Returns
    -------
    tuple of (plotly.graph_objects.Figure or None, str)
        (figure, status_message).  Figure is None when data is insufficient;
        the status_message describes the reason.
    """
    subset = df[(df['school_year'] == school_year) &
                (df['grade_level'] == grade) &
                (df['strand'] == strand)].copy()

    if len(subset) == 0: return None, "No data available for the selected filters."

    # Pivot to a wide-form student × subject matrix for correlation computation.
    pivot_df = subset.pivot_table(index='student sis', columns='course', values='numeric_grade')

    # Require at least five data points per subject to compute a meaningful
    # correlation coefficient.
    pivot_df = pivot_df.dropna(axis=1, thresh=5) 
    
    if len(pivot_df.columns) < 2: return None, "Not enough subjects to calculate correlations (need at least 2)."

    corr_matrix = pivot_df.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Purpor",
        zmin=-1, zmax=1,
        labels=dict(x="Subject", y="Subject", color="Correlation"),
        title=f"Correlation Matrix: {strand} - Grade {grade} ({school_year})"
    )

    fig.update_layout(
        height=700,
        xaxis_tickangle=-45,
        margin=dict(b=150)  # extra bottom margin for angled x-axis labels
    )

    return fig, "Success"
    
# -----------------------------------------------------------------------------
# 5. STUDENT PROFILE ANALYSIS
# -----------------------------------------------------------------------------

def get_student_kpis(df, student_sis):
    """Compute summary KPIs for a single student across their full transcript.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.
    student_sis : str
        Student Information System identifier.

    Returns
    -------
    dict
        Keys: Name, ID, Strand, Section, Cumulative GPA, Total Subjects Taken,
        Highest Grade, Lowest Grade, Latest Full Term, Latest School Year.
        Returns an empty dict if the student is not found.
    """
    student_data = df[df['student sis'] == student_sis].copy()
    if student_data.empty: return {}

    gpa = student_data['numeric_grade'].mean()
    total_subjects = student_data['course'].nunique()

    # Sort descending to reliably locate the most recent record regardless
    # of how the categorical full_term column interacts with school_year.
    latest_record = student_data.sort_values(['school_year', 'full_term'], ascending=False).iloc[0]

    kpis = {
        'Name': latest_record['student name'],
        'ID': student_sis,
        'Strand': latest_record['strand'],
        'Section': latest_record['section_name'],
        'Cumulative GPA': round(gpa, 2),
        'Total Subjects Taken': total_subjects,
        'Highest Grade': student_data['numeric_grade'].max(),
        'Lowest Grade': student_data['numeric_grade'].min(),
        'Latest Full Term': latest_record['full_term'],
        'Latest School Year': latest_record['school_year']
    }
    return kpis

def calculate_class_standing(df, student_sis):
    """Compute the student's GPA percentile rank within their most recent cohort.

    The reference cohort is defined as all students sharing the same strand,
    grade level, and school year as the student's most recent record.
    Percentile is calculated as the proportion of cohort GPAs at or below
    the student's GPA, expressed as a percentage (0–100).

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.
    student_sis : str
        Student Information System identifier.

    Returns
    -------
    str
        Formatted percentile string (e.g., '72.4th Percentile'), or 'N/A'
        if the student is not found or the cohort is too small (< 2).
    """
    student_data = df[df['student sis'] == student_sis]
    if student_data.empty: return "N/A"

    latest_record = student_data.sort_values(['school_year', 'full_term'], ascending=False).iloc[0]
    latest_strand = latest_record['strand']
    latest_grade  = latest_record['grade_level']
    latest_year   = latest_record['school_year']

    cohort_df = df[(df['strand'] == latest_strand) &
                   (df['grade_level'] == latest_grade) &
                   (df['school_year'] == latest_year)]

    cohort_gpas = cohort_df.groupby('student sis')['numeric_grade'].mean().dropna()
    student_gpa = student_data['numeric_grade'].mean()

    # Proportion of cohort at or below this student's GPA, as a percentage.
    percentile = (cohort_gpas <= student_gpa).sum() / len(cohort_gpas) * 100

    if len(cohort_gpas) < 2: return "N/A (Cohort too small)"
    return f"{percentile:.1f}th Percentile"

def plot_growth_curve(df, student_sis):
    """Plot a student's semester GWA trajectory against all four strand baselines.

    The student's mean grade per academic term (G11-S1 through G12-S2) is
    rendered as a bold primary trace.  Four dotted reference traces overlay
    the strand-level cohort means for the same term and school year, providing
    the contextual baseline against which the student's trajectory is read.

    connectgaps=True is applied to both the student trace and strand traces so
    that an unbroken line is drawn for students mid-curriculum whose transcript
    does not yet span all four terms.

    The strand cohort means are computed dynamically per term by matching on
    strand, grade_level, school_year, and full_term — ensuring that the
    reference baseline reflects the actual cohort the student was enrolled with,
    not a historical average.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.
    student_sis : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # 1. Get Student Data
    student_data = df[df['student sis'] == student_sis].copy()
    if student_data.empty: return go.Figure(go.Layout(title='No Data for Student'))

    term_order = ['G11-S1', 'G11-S2', 'G12-S1', 'G12-S2']

    # Aggregate student's mean grade per term, carrying school_year and
    # grade_level forward so strand baselines can be matched precisely.
    term_gpa = student_data.groupby('full_term').agg({
        'numeric_grade': 'mean',
        'school_year': 'max',
        'grade_level': 'max'
    }).reset_index()

    term_gpa['sort_order'] = term_gpa['full_term'].apply(lambda x: term_order.index(x) if x in term_order else 99)
    term_gpa = term_gpa.sort_values('sort_order').drop(columns='sort_order')

    # Compute cohort mean for each strand at each term the student attended.
    strands_to_plot = ['STEM', 'ABM', 'HUMSS', 'GAS']
    strand_data = {s: [] for s in strands_to_plot}

    for _, row in term_gpa.iterrows():
        year  = row['school_year']
        grade = row['grade_level']
        term  = row['full_term']

        for s in strands_to_plot:
            cohort = df[
                (df['strand'] == s) &
                (df['grade_level'] == grade) &
                (df['school_year'] == year) &
                (df['full_term'] == term)
            ]
            strand_data[s].append(cohort['numeric_grade'].mean() if not cohort.empty else None)

    fig = go.Figure()

    # Colour palette consistent with the spider chart (student_color = Bold[0]).
    colors = px.colors.qualitative.Bold
    student_color = colors[0]
    strand_colors = {
        'STEM': colors[2],
        'ABM':  colors[3],
        'HUMSS': colors[4],
        'GAS':  colors[5]
    }

    for s in strands_to_plot:
        fig.add_trace(go.Scatter(
            x=term_gpa['full_term'],
            y=strand_data[s],
            mode='lines+markers',
            name=f'{s} Avg',
            line=dict(color=strand_colors.get(s, 'grey'), width=2, dash='dot'),
            marker=dict(size=6, symbol='circle-open'),
            connectgaps=True,
            hovertemplate=f"<b>%{{x}}</b><br>{s} Avg: %{{y:.2f}}<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=term_gpa['full_term'],
        y=term_gpa['numeric_grade'],
        mode='lines+markers',
        name='Student (You)',
        line=dict(color=student_color, width=3),
        marker=dict(size=12, color=student_color, line=dict(width=2, color='white')),
        connectgaps=True,
        hovertemplate="<b>%{x}</b><br>Student Avg: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Academic Term",
        yaxis_title="Term Average Grade",
        yaxis_range=[59, 104],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig

def get_subject_performance_vs_peer(df, student_sis):
    """Compare the student's grade in each subject against their peer group mean.

    Three-tier peer matching strategy (applied in priority order):

    Priority 1 — Batchmates: students in the same course, strand, grade level,
        and school year.  This is the most contextually appropriate baseline
        and is used whenever available.

    Priority 2 — Historical same-grade cohort: all students who took the same
        course in the same strand and grade level across all school years.
        Applied when the exact batchmate cohort is unavailable (e.g., a new
        subject added after AY 2021).

    Priority 3 — Broad strand cohort: all students who took the same course in
        the same strand, regardless of grade level or year.  Applied for edge
        cases such as a Grade 12 student who took a Grade 11 subject in a
        non-standard term.

    Peer Average is returned as NaN (not coerced to a default) when all three
    tiers fail, allowing the plotting functions to handle missing data
    explicitly.  Duplicate subject entries (a student who retook a course) are
    resolved by keeping the most recent school year only.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus.
    student_sis : str

    Returns
    -------
    pandas.DataFrame
        Columns: Course, Grade Level, School Year, Student Grade,
        Peer Average, Difference, Strand.  One row per unique course taken,
        sorted descending by Student Grade.
    """
    student_data = df[df['student sis'] == student_sis].copy()
    if student_data.empty: return pd.DataFrame()

    comparison_list = []

    for index, row in student_data.iterrows():
        course = row['course']
        strand = row['strand']
        grade  = row['grade_level']
        year   = row['school_year']
        student_grade = row['numeric_grade']

        # Priority 1: exact batchmate cohort.
        peer_group = df[(df['course'] == course) &
                        (df['strand'] == strand) &
                        (df['grade_level'] == grade) &
                        (df['school_year'] == year)]
        peer_avg = peer_group['numeric_grade'].mean()

        # Priority 2: historical same strand/grade, all years.
        if pd.isna(peer_avg):
            peer_group = df[(df['course'] == course) &
                            (df['strand'] == strand) &
                            (df['grade_level'] == grade)]
            peer_avg = peer_group['numeric_grade'].mean()

        # Priority 3: broad strand, any grade, all years.
        if pd.isna(peer_avg):
            peer_group = df[(df['course'] == course) &
                            (df['strand'] == strand)]
            peer_avg = peer_group['numeric_grade'].mean()

        comparison_list.append({
            'Course': course,
            'Grade Level': grade,
            'School Year': year,
            'Student Grade': round(student_grade, 1),
            # NaN is preserved (not coerced to a default) so the plotting
            # functions can distinguish "no peer data" from a low peer mean.
            'Peer Average': round(peer_avg, 1) if pd.notna(peer_avg) else np.nan,
            'Difference': round(student_grade - peer_avg, 1) if pd.notna(peer_avg) else 0,
            'Strand': strand
        })

    comparison_df = pd.DataFrame(comparison_list).sort_values('Student Grade', ascending=False)

    # Where a student retook a course, keep only the most recent grade.
    comparison_df = comparison_df.sort_values(['Course', 'School Year'], ascending=[True, False]).drop_duplicates(subset=['Course'], keep='first')

    return comparison_df.sort_values('Student Grade', ascending=False)

def plot_subject_comparison_dumbbell(comparison_df):
    """Render a dumbbell plot comparing student grades to peer averages across all subjects.

    Each row represents one subject.  Two dots are plotted per row — the
    student's grade (primary colour) and the peer average (secondary colour) —
    connected by a grey horizontal line whose length encodes the gap magnitude.
    Subjects are sorted ascending by student grade, placing the lowest-
    performing subjects at the top of the chart for immediate visibility.

    The alternating above/below-peer pattern across subjects (the visual
    'zigzag') allows coordinators to distinguish genuine student weaknesses
    from subjects where the gap is attributable to inherent subject difficulty.

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        Output of get_subject_performance_vs_peer().

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if comparison_df.empty: return go.Figure(go.Layout(title='No Comparison Data'))

    comparison_df = comparison_df.sort_values('Student Grade', ascending=True)

    colors = px.colors.qualitative.Bold
    student_color = colors[0]
    peer_color    = colors[1]

    fig = go.Figure()

    # Grey connector lines — rendered first so they appear below the dots.
    fig.add_trace(go.Scatter(
        x=comparison_df['Peer Average'],
        y=comparison_df['Course'],
        mode='lines',
        line=dict(color='grey', width=1.5),
        showlegend=False,
        hoverinfo='none'
    ))

    # Student grade dots.
    fig.add_trace(go.Scatter(
        x=comparison_df['Student Grade'],
        y=comparison_df['Course'],
        mode='markers',
        name='Student Grade',
        marker=dict(size=10, color=student_color),
        hovertemplate="<b>Student:</b> %{x}<br>Course: %{y}<extra></extra>"
    ))

    # Peer average dots.
    fig.add_trace(go.Scatter(
        x=comparison_df['Peer Average'],
        y=comparison_df['Course'],
        mode='markers',
        name='Peer Average',
        marker=dict(size=8, color=peer_color),
        hovertemplate="<b>Peer Avg:</b> %{x}<br>Course: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title="Strength vs. Weakness: Student Grade vs. Peer Average",
        xaxis_title="Grade",
        yaxis_title="Course",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=150, r=20, t=50, b=50) 
    )
    fig.update_xaxes(range=[60, 100])
    return fig

def plot_spider_graph(comparison_df):
    """Render a radial spider chart comparing a student's subject grades to peer averages.

    Two filled polygons are drawn on a go.Scatterpolar axis:
      • Peer Average polygon (drawn first, semi-transparent fill) — the
        cohort reference shape.
      • Student Grade polygon (drawn second, higher opacity) — the student's
        actual performance profile.

    The geometric difference between the two polygons encodes the student's
    relative strengths and weaknesses: bulges outward indicate above-peer
    performance; retreats inward indicate below-peer performance.

    Peer Average smart imputation: where the three-tier peer-matching logic in
    get_subject_performance_vs_peer() returns NaN for a subject, the cohort
    median is substituted to maintain polygon closure.  The hover tooltip
    still displays 'N/A' for those subjects so the imputation is transparent
    to the user.

    Subject axis labels are truncated via generate_subject_code() to prevent
    overlap; full names are always available via hover (customdata).

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        Output of get_subject_performance_vs_peer().

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if comparison_df.empty: return go.Figure(go.Layout(title='No Comparison Data'))

    plot_df = comparison_df.copy()

    # Truncated axis labels for display; full names served via customdata hover.
    plot_df['Display_Course_Name'] = plot_df['Course'].apply(generate_subject_code)

    plot_df = plot_df.sort_values('Student Grade', ascending=False)

    # Substitute cohort median where peer data is unavailable to maintain
    # a closed polygon.  Hover text explicitly marks these as 'N/A'.
    peer_median = plot_df['Peer Average'].median()
    if pd.isna(peer_median): peer_median = 65

    plot_df['Peer Average Filled'] = plot_df['Peer Average'].fillna(peer_median)

    plot_df['Peer_Hover_String'] = plot_df['Peer Average'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

    categories = plot_df['Display_Course_Name'].tolist()

    # customdata packs [Full Course Name, Peer Hover String] for each point
    # so hover templates can show the full name regardless of axis truncation.
    plot_df['Custom_Hover_Data'] = plot_df[['Course', 'Peer_Hover_String']].values.tolist()

    colors = px.colors.qualitative.Bold
    student_color = colors[0]
    peer_color    = colors[1]

    fig = go.Figure()

    # Peer polygon — drawn first so the student polygon renders on top.
    fig.add_trace(go.Scatterpolar(
        r=plot_df['Peer Average Filled'].tolist(), 
        theta=categories,
        fill='toself',
        name='Peer Average',
        marker=dict(size=12, color=peer_color, line=dict(width=1, color='white')), 
        line=dict(color=peer_color),
        opacity=0.5,
        hovertemplate="<b>%{customdata[0]}</b><br>Peer Average: %{customdata[1]}<extra></extra>", 
        customdata=plot_df['Custom_Hover_Data'], 
        mode='lines+markers',
        hoveron='points'
    ))

    # Student polygon — drawn second, rendered on top.
    fig.add_trace(go.Scatterpolar(
        r=plot_df['Student Grade'].tolist(),
        theta=categories,
        fill='toself',
        name='Student Grade',
        marker=dict(size=12, color=student_color, line=dict(width=1, color='white')),
        line=dict(color=student_color),
        opacity=0.7,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "Student Grade: %{r}<br>" +
            "Peer Average: %{customdata[1]}<extra></extra>"
        ),
        customdata=plot_df['Custom_Hover_Data'],
        mode='lines+markers',
        hoveron='points'
    ))

    # Radial axis range 60–104 covers the full grade scale with a small margin
    # above 100 so perfect grades do not clip at the polygon edge.
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                type='category',
                categoryorder='array',
                categoryarray=categories
            ),
            radialaxis=dict(
                visible=True,
                range=[60, 104],
                dtick=5,
                angle=90,
                tickangle=0,
                showline=False,
                tickfont=dict(color='black', size=10),
                tickcolor='darkgray',
                gridcolor='lightgray',
                gridwidth=1
            )
        ),
        showlegend=True
    )
    return fig

# -----------------------------------------------------------------------------
# 6. PREDICTIVE ANALYTICS ENGINE
# -----------------------------------------------------------------------------
 
# AT_RISK_THRESHOLD — institutional minimum passing grade (DepEd K-12 policy).
# Subjects where a student's predicted grade falls below this value are
# flagged for intervention.  The same constant governs both the RF classifier
# target label and the visual threshold line in the predictive charts.
AT_RISK_THRESHOLD = 80
 
# ── Shared Helpers ────────────────────────────────────────────────────────────
 
def _encode_categoricals(df_in):
    """Encode string categorical columns to integer codes for Random Forest input.

    LabelEncoder produces integer codes sorted alphabetically by category
    value.  For strand (ABM, GAS, HUMSS, STEM) the encoding is stable across
    calls on the same corpus because the full vocabulary is always present.
    For semester (S0, S1, S2) the encoding is similarly stable.

    This function also derives two numeric features used by both RF pipelines:
      • grade_int  : grade_level cast to int (11 or 12).
      • year_int   : first four characters of school_year cast to int
                     (e.g., '2024-2025' → 2024).

    Parameters
    ----------
    df_in : pandas.DataFrame
        DataFrame containing at least 'strand', 'semester', 'grade_level',
        and 'school_year' columns.

    Returns
    -------
    tuple of (pandas.DataFrame, LabelEncoder, LabelEncoder)
        Encoded DataFrame, fitted strand encoder, fitted semester encoder.
    """
    df = df_in.copy()
    le_strand = LabelEncoder()
    le_sem    = LabelEncoder()
 
    df['strand_enc']   = le_strand.fit_transform(df['strand'].fillna('STEM'))
    df['semester_enc'] = le_sem.fit_transform(df['semester'].fillna('S1'))
    df['grade_int']    = pd.to_numeric(df['grade_level'], errors='coerce').fillna(11).astype(int)
    df['year_int']     = df['school_year'].str[:4].astype(int)
 
    return df, le_strand, le_sem
 
 
# ── MACRO MODEL (Cohort-Level) ────────────────────────────────────────────────
 
def build_macro_features(df):
    """Aggregate the longitudinal corpus to cohort level and engineer prediction features.

    Transforms the student-subject-grade record corpus into a one-row-per-
    cohort-slot design matrix suitable for Random Forest training.  Each row
    represents one (strand, grade_level, course, school_year, semester)
    combination.

    Feature engineering
    -------------------
    Encoded context features:
        strand_enc, semester_enc  : integer-encoded via _encode_categoricals()
        grade_int, year_int       : integer grade level and start year

    Distributional features (computed from the raw grade distribution):
        mean_grade  : arithmetic mean (the regression target)
        std_grade   : standard deviation (grade spread)
        skewness    : scipy.stats.skew(); negative = left-skewed (ceiling effect),
                      positive = right-skewed (barrier subject)
        n_students  : unique student count (cohort size)

    Temporal lag features (same course, strand, grade across adjacent periods):
        prior_mean  : mean_grade from the immediately preceding semester
        prior_std   : std_grade from the immediately preceding semester
        Both are computed via groupby().shift(1) after sorting by year and
        semester; first-occurrence NaN values are median-imputed.

    Strand-level contextual feature:
        strand_gwa  : mean numeric_grade across ALL courses for the same
                      strand, grade_level, school_year, and semester —
                      a signal of overall strand-level performance that semester.

    Target columns
    --------------
        mean_grade : continuous target for the RF Regressor.
        at_risk    : binary flag (1 if mean_grade < AT_RISK_THRESHOLD) for
                     the RF Classifier.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus from load_and_process_data().

    Returns
    -------
    pandas.DataFrame
        One row per cohort slot with all features and targets.
        Returns an empty DataFrame if the input is empty.
    """
    if df.empty:
        return pd.DataFrame()

    # Aggregate to cohort level.
    cohort = df.groupby(
        ['school_year', 'semester', 'grade_level', 'strand', 'course']
    ).agg(
        mean_grade = ('numeric_grade', 'mean'),
        std_grade  = ('numeric_grade', 'std'),
        n_students = ('student sis', 'nunique'),
        skewness   = ('numeric_grade', lambda x: stats.skew(x.dropna())
                       if len(x.dropna()) > 2 else 0.0)
    ).reset_index().dropna(subset=['mean_grade'])

    cohort, _, _ = _encode_categoricals(cohort)

    # Sort so shift() targets the chronologically preceding semester.
    cohort = cohort.sort_values(
        ['strand', 'grade_level', 'course', 'year_int', 'semester_enc']
    )

    # Lag features: previous semester's mean and spread for the same subject.
    grp = cohort.groupby(['strand', 'grade_level', 'course'])
    cohort['prior_mean'] = grp['mean_grade'].shift(1)
    cohort['prior_std']  = grp['std_grade'].shift(1)

    # Strand GWA: mean across ALL subjects for the same strand/grade/period.
    strand_gwa = (
        df.groupby(['school_year', 'semester', 'grade_level', 'strand'])['numeric_grade']
        .mean().reset_index()
        .rename(columns={'numeric_grade': 'strand_gwa'})
    )
    cohort = cohort.merge(
        strand_gwa, on=['school_year', 'semester', 'grade_level', 'strand'], how='left'
    )

    cohort['at_risk'] = (cohort['mean_grade'] < AT_RISK_THRESHOLD).astype(int)

    # Median imputation for lag features that are NaN at first occurrence.
    for col in ['prior_mean', 'prior_std', 'std_grade', 'strand_gwa']:
        cohort[col] = cohort[col].fillna(cohort[col].median())

    return cohort.reset_index(drop=True)
 
 
# MACRO_FEATURES — ordered feature set for the cohort-level Random Forest.
# All ten features are derived by build_macro_features(); any column not
# present in the cohort DataFrame (e.g., due to an insufficient corpus) is
# silently excluded before model training.
MACRO_FEATURES = [
    'strand_enc', 'grade_int', 'semester_enc', 'year_int',
    'prior_mean', 'prior_std', 'std_grade',
    'n_students', 'skewness', 'strand_gwa'
]
 
 
def train_macro_model(df):
    """Train the cohort-level Random Forest Regressor and Classifier.

    Applies a temporal holdout strategy: all school years except the most
    recent form the training set; the most recent year is the test set.
    This mirrors the deployment scenario where the model is always predicting
    a future period it has not seen during training.

    Regressor: RandomForestRegressor (200 trees, min_samples_leaf=3) targeting
    mean_grade.

    Classifier: RandomForestClassifier (200 trees, min_samples_leaf=3,
    class_weight='balanced') targeting at_risk.  class_weight='balanced'
    prevents the majority class (on-track subjects) from dominating parameter
    updates when at-risk subjects are rare.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus from load_and_process_data().

    Returns
    -------
    tuple of (RandomForestRegressor or None, RandomForestClassifier or None,
              dict, pandas.DataFrame)
        (regressor, classifier, val_metrics, feature_importance_df).
        regressor and classifier are None if insufficient data is available;
        val_metrics contains 'error' key describing the failure.
        val_metrics keys when successful: MAE, R2, AUC, train_n, test_n.
        AUC is 'N/A' when the test set is single-class.
    """
    cohort = build_macro_features(df)
    if cohort.empty or len(cohort) < 20:
        return None, None, {'error': 'Insufficient cohort data'}, pd.DataFrame()
 
    available = [f for f in MACRO_FEATURES if f in cohort.columns]
 
    latest_year = cohort['year_int'].max()
    train = cohort[cohort['year_int'] < latest_year].dropna(subset=available)
    test  = cohort[cohort['year_int'] == latest_year].dropna(subset=available)
 
    if len(train) < 10 or len(test) < 5:
        return None, None, {'error': 'Insufficient data after temporal split'}, pd.DataFrame()
 
    X_tr, y_reg_tr = train[available], train['mean_grade']
    X_te, y_reg_te = test[available],  test['mean_grade']
    y_cls_tr, y_cls_te = train['at_risk'], test['at_risk']
 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
 
        reg = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )
        reg.fit(X_tr, y_reg_tr)
        reg_preds = reg.predict(X_te)
 
        cls = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        cls.fit(X_tr, y_cls_tr)
        cls_proba = cls.predict_proba(X_te)[:, 1]
 
    auc_val = (
        round(roc_auc_score(y_cls_te, cls_proba), 3)
        if y_cls_te.nunique() > 1 else 'N/A'
    )
    val_metrics = {
        'MAE':     round(mean_absolute_error(y_reg_te, reg_preds), 3),
        'R2':      round(r2_score(y_reg_te, reg_preds), 3),
        'AUC':     auc_val,
        'train_n': len(train),
        'test_n':  len(test)
    }
 
    fi_df = pd.DataFrame({
        'Feature':    available,
        'Importance': reg.feature_importances_
    }).sort_values('Importance', ascending=False)
 
    return reg, cls, val_metrics, fi_df
 
 
def predict_macro_outlook(reg, cls, df, strand, grade_level, next_semester, next_year_str):
    """Generate predicted mean grades and at-risk probabilities for an upcoming semester.

    For each subject in the specified strand and grade level, the most recent
    historical cohort record is used as the feature baseline.  Time-encoding
    features (year_int, semester_enc) are advanced by one period, and the
    current actual mean is promoted to the lag feature (prior_mean) for the
    projected term.

    Risk labels:
        🔴 High Risk  : at-risk probability >= 0.60
        🟡 Moderate   : 0.35 <= probability < 0.60
        🟢 On Track   : probability < 0.35

    Parameters
    ----------
    reg : RandomForestRegressor
        Trained macro regressor from train_macro_model().
    cls : RandomForestClassifier
        Trained macro classifier from train_macro_model().
    df : pandas.DataFrame
        Full longitudinal corpus.
    strand, grade_level : str
        Target cohort identifiers.
    next_semester : str
        'S1' or 'S2'
    next_year_str : str
        School year string (e.g. '2025-2026'); only the first 4 characters
        are used to derive year_int.

    Returns
    -------
    pandas.DataFrame
        Columns: course, predicted_mean, risk_probability, risk_label,
        prior_mean, strand, grade_level.  Sorted ascending by predicted_mean
        (highest-risk subjects appear first).  Empty if reg is None or the
        cohort has no historical data.
    """
    if reg is None:
        return pd.DataFrame()

    cohort = build_macro_features(df)
    subset = cohort[
        (cohort['strand'] == strand) &
        (cohort['grade_level'] == grade_level)
    ].sort_values('year_int', ascending=False)

    if subset.empty:
        return pd.DataFrame()

    # Most recent record per subject serves as the feature baseline.
    latest = subset.drop_duplicates(subset=['course'], keep='first').copy()

    # Advance time-encoding to the target prediction period.
    latest['year_int']     = int(next_year_str[:4])
    latest['semester_enc'] = {'S1': 1, 'S2': 2}.get(next_semester, 1)
    # Current actual mean becomes the lag feature for the projected semester.
    latest['prior_mean']   = latest['mean_grade']
    latest['prior_std']    = latest['std_grade']

    available = [f for f in MACRO_FEATURES if f in latest.columns]
    X = latest[available].fillna(latest[available].median())
 
    latest['predicted_mean']    = reg.predict(X).round(2)
    latest['risk_probability']  = cls.predict_proba(X)[:, 1].round(3)
    latest['risk_label'] = latest['risk_probability'].apply(
        lambda p: '🔴 High Risk' if p >= 0.6
        else ('🟡 Moderate' if p >= 0.35 else '🟢 On Track')
    )
 
    return latest[[
        'course', 'predicted_mean', 'risk_probability',
        'risk_label', 'prior_mean', 'strand', 'grade_level'
    ]].sort_values('predicted_mean', ascending=True).reset_index(drop=True)
 
 
# ── MICRO MODEL (Student-Level) ───────────────────────────────────────────────
 
def build_micro_features(df):
    """Build a student-subject record feature matrix for the micro-level RF pipeline.

    Each row corresponds to one student-subject-semester grade record, enriched
    with eleven engineered features organised into three theoretical constructs:

    Academic trajectory features (student-level, time-ordered):
        prior_gwa      : student's mean grade across all subjects in the
                         immediately preceding academic term (groupby shift(1)
                         on term GWA sorted by year_int and term_order).
        cumulative_gwa : student's expanding mean across all prior records
                         (shift(1) of the expanding window ensures the target
                         record is not included in its own baseline).
        gwa_trend      : linear slope of the student's per-term GWA sequence
                         (computed via np.polyfit; returns 0.0 for students
                         with only one observed term).

    Subject difficulty features (strand × grade × course aggregates):
        subj_hist_mean : historical mean grade for this subject in this strand
                         and grade level across all school years.
        subj_hist_std  : historical standard deviation.
        subj_skewness  : scipy.stats.skew() of the historical grade distribution
                         (0.0 if fewer than three records).

    Contextual baseline feature:
        peer_mean      : mean grade for this subject in this exact strand,
                         grade level, semester, and school year — the most
                         precise available cohort reference.

    All NaN values in the above features are median-imputed before the
    feature matrix is returned; the at_risk binary target is computed last.

    Sort column selection: if year_int and term_order are both present (they
    always are after _encode_categoricals and the term_order map), sorting
    uses ['student sis', 'year_int', 'term_order'].  A safer fallback to
    ['student sis', 'school_year', 'full_term'] is applied otherwise to prevent
    KeyError on edge-case corpus states.

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus from load_and_process_data().

    Returns
    -------
    pandas.DataFrame
        One row per student-subject record with all MICRO_FEATURES columns,
        plus 'at_risk' binary target.  Returns an empty DataFrame if the input
        is empty.
    """
    if df.empty:
        return pd.DataFrame()

    feat = df.copy()
    feat, _, _ = _encode_categoricals(feat)
    feat['term_order'] = feat['full_term'].map(
        {'G11-S1': 1, 'G11-S2': 2, 'G12-S1': 3, 'G12-S2': 4}
    ).astype(float).fillna(0).astype(int)

    # Sort column selection — uses year_int+term_order when available for
    # precise chronological ordering; falls back to school_year+full_term.
    if 'year_int' in feat.columns and 'term_order' in feat.columns:
        sort_cols = ['student sis', 'year_int', 'term_order']
    else:
        sort_cols = ['student sis', 'school_year', 'full_term']

    # Include sort columns in the groupby key set to preserve them after
    # aggregation (pandas drops non-aggregated columns in groupby results).
    group_keys = list(dict.fromkeys(['student sis', 'school_year', 'full_term'] + sort_cols))

    term_gwa = (
        feat.groupby(group_keys)['numeric_grade']
        .mean().reset_index().rename(columns={'numeric_grade': 'term_gwa'})
        .sort_values(sort_cols)
    )

    term_gwa['prior_gwa'] = term_gwa.groupby('student sis')['term_gwa'].shift(1)

    feat = feat.merge(
        term_gwa[['student sis', 'school_year', 'full_term', 'prior_gwa']],
        on=['student sis', 'school_year', 'full_term'], how='left'
    )
 
    # ── Cumulative GWA ──────────────────────────────────────────────────────
    feat = feat.sort_values(sort_cols)
    feat['cumulative_gwa'] = (
        feat.groupby('student sis')['numeric_grade']
        .expanding().mean().shift(1)
        .reset_index(level=0, drop=True)
    )

    def _slope(series):
        """Return the least-squares linear slope of a term-GWA sequence."""
        vals = series.dropna().values
        if len(vals) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(vals)), vals, 1)[0])

    gwa_trends = (
        term_gwa.groupby('student sis')['term_gwa']
        .apply(_slope).reset_index()
        .rename(columns={'term_gwa': 'gwa_trend'})
    )
    feat = feat.merge(gwa_trends, on='student sis', how='left')

    # Subject difficulty baseline: historical mean, std, and skewness for
    # each strand × grade_level × course combination across all school years.
    subj_hist = (
        df.groupby(['strand', 'grade_level', 'course'])['numeric_grade']
        .agg(['mean', 'std']).reset_index()
        .rename(columns={'mean': 'subj_hist_mean', 'std': 'subj_hist_std'})
    )
    subj_skew = (
        df.groupby(['strand', 'grade_level', 'course'])['numeric_grade']
        .apply(lambda x: stats.skew(x.dropna()) if len(x.dropna()) > 2 else 0.0)
        .reset_index().rename(columns={'numeric_grade': 'subj_skewness'})
    )
    feat = feat.merge(subj_hist, on=['strand', 'grade_level', 'course'], how='left')
    feat = feat.merge(subj_skew, on=['strand', 'grade_level', 'course'], how='left')

    # Peer mean: exact cohort reference for this subject, strand, grade,
    # semester, and school year.
    peer_mean = (
        df.groupby(['school_year', 'semester', 'grade_level', 'strand', 'course'])
        ['numeric_grade'].mean().reset_index()
        .rename(columns={'numeric_grade': 'peer_mean'})
    )
    feat = feat.merge(
        peer_mean,
        on=['school_year', 'semester', 'grade_level', 'strand', 'course'],
        how='left'
    )

    # Median imputation for any feature that has NaN due to a first-occurrence
    # or sparse cohort condition.
    for col in ['prior_gwa', 'cumulative_gwa', 'gwa_trend',
                'subj_hist_std', 'subj_skewness', 'peer_mean']:
        feat[col] = feat[col].fillna(feat[col].median())

    feat['at_risk'] = (feat['numeric_grade'] < AT_RISK_THRESHOLD).astype(int)

    return feat.reset_index(drop=True)
 
 
# MICRO_FEATURES — ordered feature set for the student-subject Random Forest.
# Eleven features are engineered per student-subject record by
# build_micro_features().  The set encodes three theoretical constructs:
#   • Academic trajectory  : prior_gwa, cumulative_gwa, gwa_trend
#   • Subject difficulty   : subj_hist_mean, subj_hist_std, subj_skewness
#   • Contextual baseline  : peer_mean, strand_enc, grade_int, semester_enc, term_order
MICRO_FEATURES = [
    'strand_enc', 'grade_int', 'semester_enc', 'term_order',
    'prior_gwa', 'cumulative_gwa', 'gwa_trend',
    'subj_hist_mean', 'subj_hist_std', 'subj_skewness', 'peer_mean'
]
 
 
def train_micro_model(df):
    """Train the student-subject Random Forest Regressor and Classifier.

    Operates at individual student-subject record granularity using the eleven
    features engineered by build_micro_features().  Applies the same temporal
    holdout strategy as train_macro_model(): all years except the most recent
    form the training set; the most recent year is the test set.

    Regressor: RandomForestRegressor (300 trees, min_samples_leaf=5).
    A higher tree count and larger leaf size are used compared to the macro
    model because the micro dataset is substantially larger and more noisy.

    Classifier: RandomForestClassifier (300 trees, min_samples_leaf=5,
    class_weight='balanced') targeting at_risk (grade < AT_RISK_THRESHOLD).

    Parameters
    ----------
    df : pandas.DataFrame
        Full longitudinal corpus from load_and_process_data().

    Returns
    -------
    tuple of (RandomForestRegressor or None, RandomForestClassifier or None,
              dict, pandas.DataFrame)
        Same structure as train_macro_model().
    """
    feat = build_micro_features(df)
    if feat.empty or len(feat) < 50:
        return None, None, {'error': 'Insufficient student data'}, pd.DataFrame()
 
    available = [f for f in MICRO_FEATURES if f in feat.columns]
    clean     = feat.dropna(subset=available + ['numeric_grade'])
 
    latest_year = clean['year_int'].max()
    train = clean[clean['year_int'] < latest_year]
    test  = clean[clean['year_int'] == latest_year]
 
    if len(train) < 20 or len(test) < 10:
        return None, None, {'error': 'Insufficient data after temporal split'}, pd.DataFrame()
 
    X_tr, y_reg_tr = train[available], train['numeric_grade']
    X_te, y_reg_te = test[available],  test['numeric_grade']
    y_cls_tr, y_cls_te = train['at_risk'], test['at_risk']
 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
 
        reg = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        reg.fit(X_tr, y_reg_tr)
        reg_preds = reg.predict(X_te)
 
        cls = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        cls.fit(X_tr, y_cls_tr)
        cls_proba = cls.predict_proba(X_te)[:, 1]
 
    auc_val = (
        round(roc_auc_score(y_cls_te, cls_proba), 3)
        if y_cls_te.nunique() > 1 else 'N/A'
    )
    val_metrics = {
        'MAE':     round(mean_absolute_error(y_reg_te, reg_preds), 3),
        'R2':      round(r2_score(y_reg_te, reg_preds), 3),
        'AUC':     auc_val,
        'train_n': len(train),
        'test_n':  len(test)
    }
 
    fi_df = pd.DataFrame({
        'Feature':    available,
        'Importance': reg.feature_importances_
    }).sort_values('Importance', ascending=False)

    return reg, cls, val_metrics, fi_df


def predict_student_outlook(micro_reg, micro_cls, df, student_sis):
    """Predict grade performance for a student's most recently enrolled semester.

    Identifies the student's latest school_year and full_term, retrieves the
    corresponding rows from the micro feature matrix, and runs inference using
    the trained RF models.  Because this prediction applies to the current
    semester (where actual grades may or may not yet be posted), the output
    can be validated against whatever grades are available.

    Results are sorted descending by risk_probability so that highest-
    intervention-priority subjects appear first.

    Parameters
    ----------
    micro_reg : RandomForestRegressor
    micro_cls : RandomForestClassifier
    df : pandas.DataFrame
        Full longitudinal corpus.
    student_sis : str

    Returns
    -------
    pandas.DataFrame
        Columns: course, numeric_grade (actual if available), predicted_grade,
        risk_probability, risk_label, peer_mean.
        Returns an empty DataFrame if the student is not found or models are None.
    """
    if micro_reg is None:
        return pd.DataFrame()
 
    feat = build_micro_features(df)
    student = feat[feat['student sis'] == student_sis].copy()
    if student.empty:
        return pd.DataFrame()
 
    latest = student.sort_values(['year_int', 'term_order']).iloc[-1]
    current = student[
        (student['school_year'] == latest['school_year']) &
        (student['full_term']   == latest['full_term'])
    ].copy()
 
    available = [f for f in MICRO_FEATURES if f in current.columns]
    X = current[available].fillna(current[available].median())
 
    current = current.copy()
    current['predicted_grade']   = micro_reg.predict(X).round(2)
    current['risk_probability']  = micro_cls.predict_proba(X)[:, 1].round(3)
    current['risk_label'] = current['risk_probability'].apply(
        lambda p: '🔴 High Risk' if p >= 0.6
        else ('🟡 Moderate' if p >= 0.35 else '🟢 On Track')
    )
 
    return current[[
        'course', 'numeric_grade', 'predicted_grade',
        'risk_probability', 'risk_label', 'peer_mean'
    ]].sort_values('risk_probability', ascending=False).reset_index(drop=True)

# ── CURRICULUM MAP & FUTURE FORECAST ─────────────────────────────────────────

def extract_curriculum_map(df, threshold=0.4, recent_years=3):
    """
    Builds a strand-aware curriculum roadmap from the past N school years.

    Key design decisions:
    - Uses `section sis` (via `process_section_info`) as the authoritative
      source of strand AND grade_level, matching `load_and_process_data` exactly.
    - Uses `term sis` for semester recovery (S1/S2), with `.iloc[:, 0]` for
      pandas-version-safe column access.
    - Restricts to the most recent `recent_years` school years so deprecated
      or renamed subjects don't pollute the roadmap.
    - A subject is included in a strand/grade/semester slot only if it appeared
      for >= `threshold` fraction of students in that cohort.

    Returns DataFrame with columns:
        strand | grade_level | semester | course | rate
    """
    temp_df = df.copy()

    # 1. Re-derive strand and grade_level from section sis (authoritative source)
    if 'section sis' in temp_df.columns:
        extracted = temp_df['section sis'].apply(process_section_info)
        temp_df['strand']      = [x[0] for x in extracted]
        temp_df['grade_level'] = [x[1] for x in extracted]

    # 2. Recover semester from term sis using .iloc[:, 0] (pandas-safe)
    if 'term sis' in temp_df.columns:
        recovered = (
            temp_df['term sis']
            .str.extract(r'_(S[12]|[12])(?:_|$)')
            .iloc[:, 0]
            .replace({'1': 'S1', '2': 'S2'})
            .fillna('S1')
        )
        temp_df['semester'] = recovered.where(recovered.notna(), temp_df.get('semester', pd.NA))

    # 3. Standardize all key columns
    if 'course' in temp_df.columns:
        temp_df['course'] = temp_df['course'].str.strip().replace(SUBJECT_NAME_MAPPING)

    for col in ['strand', 'grade_level', 'semester']:
        temp_df[col] = temp_df[col].astype(str).str.strip().str.upper()

    # 4. Drop rows with missing keys
    temp_df = temp_df.dropna(subset=['strand', 'grade_level', 'semester', 'course', 'student sis'])
    temp_df = temp_df[
        (temp_df['strand'].isin(['STEM', 'ABM', 'HUMSS', 'GAS'])) &
        (temp_df['grade_level'].isin(['11', '12'])) &
        (temp_df['semester'].isin(['S1', 'S2']))
    ]

    # 5. Filter to recent N school years
    if 'school_year' in temp_df.columns:
        recent = sorted(temp_df['school_year'].dropna().unique())[-recent_years:]
        temp_df = temp_df[temp_df['school_year'].isin(recent)]

    if temp_df.empty:
        return pd.DataFrame()

    # 6. Build roadmap: subject must appear in >= threshold of students per cohort
    group_counts = (
        temp_df.groupby(['strand', 'grade_level', 'semester'])['student sis']
        .nunique().reset_index(name='total')
    )
    subj_counts = (
        temp_df.groupby(['strand', 'grade_level', 'semester', 'course'])['student sis']
        .nunique().reset_index(name='count')
    )
    curric = subj_counts.merge(group_counts, on=['strand', 'grade_level', 'semester'])
    curric['rate'] = curric['count'] / curric['total']

    result = curric[curric['rate'] >= threshold][
        ['strand', 'grade_level', 'semester', 'course', 'rate']
    ].reset_index(drop=True)

    # 7. Fallback: lower threshold if result is sparse
    if result.empty:
        curric2 = curric[curric['rate'] >= 0.1][
            ['strand', 'grade_level', 'semester', 'course', 'rate']
        ].reset_index(drop=True)
        return curric2

    return result


def predict_future_performance(micro_reg, micro_cls, df, student_sis):
    """
    Predicts grades for subjects the student has NOT yet taken,
    based on what their strand curriculum expects in future terms.

    Logic:
    1. Determine the student's strand and their latest completed term_order.
    2. Only predict terms AFTER their current position
       (e.g. G11-S1 student → predict G11-S2, G12-S1, G12-S2).
    3. Source the curriculum from extract_curriculum_map (recent 3 years,
       section sis-derived strand/grade).
    4. Skip any subject already in the student's transcript (case-insensitive,
       matched by grade_level + course to avoid skipping G12 repeats of G11 subjects
       with the same name).
    5. Build synthetic feature rows and run the trained RF models.
    """
    if micro_reg is None or df.empty:
        return pd.DataFrame()

    # ── Step 1: Re-derive semester locally (safe extract) ───────────────────
    local_df = df.copy()
    if 'term sis' in local_df.columns:
        recovered = (
            local_df['term sis']
            .str.extract(r'_(S[12]|[12])(?:_|$)')
            .iloc[:, 0]
            .replace({'1': 'S1', '2': 'S2'})
            .fillna('S1')
        )
        local_df['semester'] = recovered.where(recovered.notna(), local_df.get('semester', pd.NA))

    # ── Step 2: Build micro features & locate student ───────────────────────
    feat = build_micro_features(local_df)
    student_feat = feat[feat['student sis'] == student_sis].copy()
    if student_feat.empty:
        return pd.DataFrame()

    # ── Step 3: Determine student's strand and latest completed term ─────────
    student_strand = str(student_feat['strand'].iloc[0]).strip().upper()
    latest_term_order = int(student_feat['term_order'].max())

    # term_order mapping: G11-S1=1, G11-S2=2, G12-S1=3, G12-S2=4
    TERM_MAP = {'11-S1': 1, '11-S2': 2, '12-S1': 3, '12-S2': 4}
    FUTURE_TERMS = {k: v for k, v in TERM_MAP.items() if v > latest_term_order}

    if not FUTURE_TERMS:
        # Student has completed all terms — nothing to predict
        return pd.DataFrame()

    # ── Step 4: Build set of already-taken courses per grade level ───────────
    # Key: (grade_level_str, course_upper) — prevents skipping a legitimately
    # distinct G12 subject that shares a name with a G11 subject.
    taken_set = set(
        zip(
            student_feat['grade_level'].astype(str).str.strip(),
            student_feat['semester'].astype(str).str.strip().str.upper(),
            student_feat['course'].str.strip().str.upper()
        )
    )

    # ── Step 5: Get the strand curriculum (recent 3 years) ──────────────────
    curric = extract_curriculum_map(local_df, recent_years=3)
    strand_curric = curric[curric['strand'] == student_strand]

    if strand_curric.empty:
        # Fallback: lower threshold
        curric = extract_curriculum_map(local_df, threshold=0.1, recent_years=3)
        strand_curric = curric[curric['strand'] == student_strand]

    if strand_curric.empty:
        return pd.DataFrame()

    # ── Step 6: Build synthetic feature rows for future subjects ────────────
    current_avg  = student_feat['numeric_grade'].mean()
    latest_row   = student_feat.sort_values('term_order').iloc[-1]

    rows = []
    for _, item in strand_curric.iterrows():
        grade_lvl = str(item['grade_level']).strip()
        semester  = str(item['semester']).strip().upper()
        term_key  = f"{grade_lvl}-{semester}"
        t_order   = TERM_MAP.get(term_key, 0)

        # Only predict subjects in FUTURE terms
        if t_order not in FUTURE_TERMS.values():
            continue

        course_name = str(item['course']).strip()
        course_upper = course_name.upper()

        # Skip if already taken at this grade level
        if (grade_lvl, semester, course_upper) in taken_set:
            continue

        # Subject difficulty baseline from historical records
        subj_raw = local_df[local_df['course'] == course_name]['numeric_grade'].dropna()
        _mean = subj_raw.mean() if pd.notna(subj_raw.mean()) else 85.0
        _std  = subj_raw.std()  if pd.notna(subj_raw.std())  else 5.0
        _skew = stats.skew(subj_raw) if len(subj_raw) > 2 else 0.0

        rows.append({
            'course':        course_name,
            'full_term':     f"G{term_key}",
            'term_order':    t_order,
            'strand_enc':    latest_row['strand_enc'],
            'grade_int':     int(grade_lvl),
            'semester_enc':  1 if semester == 'S1' else 2,
            'prior_gwa':     current_avg,
            'cumulative_gwa': current_avg,
            'gwa_trend':     float(latest_row.get('gwa_trend', 0.0)),
            'subj_hist_mean': _mean,
            'subj_hist_std':  _std,
            'subj_skewness':  _skew,
            'peer_mean':      _mean,
        })

    if not rows:
        return pd.DataFrame()

    future_df = pd.DataFrame(rows).drop_duplicates(subset=['course'])

    # ── Step 7: Predict ──────────────────────────────────────────────────────
    X = future_df[MICRO_FEATURES].fillna(85.0)
    future_df['predicted_grade']  = micro_reg.predict(X).round(2)
    future_df['risk_probability'] = micro_cls.predict_proba(X)[:, 1].round(3)
    future_df['numeric_grade']    = np.nan  # placeholder for dumbbell chart
    future_df['risk_label'] = future_df['risk_probability'].apply(
        lambda p: '🔴 High Risk' if p >= 0.6 else ('🟡 Moderate' if p >= 0.35 else '🟢 On Track')
    )

    return future_df.sort_values(['term_order', 'course']).reset_index(drop=True)
 
# ── Predictive Visualizations ─────────────────────────────────────────────────
 
# _FEATURE_LABELS — human-readable display names for feature importance charts.
# Keys are the internal column names produced by build_macro_features() and
# build_micro_features(); values are the labels shown in the Plotly figure.
_FEATURE_LABELS = {
    'prior_gwa':       'Prior Term GWA',
    'cumulative_gwa':  'Cumulative GWA',
    'gwa_trend':       'GWA Trend (Slope)',
    'subj_hist_mean':  'Subject Historical Mean',
    'subj_hist_std':   'Subject Grade Spread (SD)',
    'subj_skewness':   'Subject Distribution Skewness',
    'peer_mean':       'Peer Cohort Mean',
    'strand_enc':      'Strand',
    'grade_int':       'Grade Level',
    'semester_enc':    'Semester',
    'term_order':      'Term Position',
    'prior_mean':      'Prior Semester Mean (Cohort)',
    'prior_std':       'Prior Semester Spread (Cohort)',
    'std_grade':       'Grade Std Dev (Cohort)',
    'n_students':      'Cohort Size',
    'skewness':        'Cohort Distribution Skewness',
    'strand_gwa':      'Strand GWA',
    'year_int':        'School Year'
}
 
# _RISK_COLORS — consistent colour encoding for the three risk categories
# used across both the macro prediction bar chart and the micro dumbbell chart.
_RISK_COLORS = {
    '🔴 High Risk': '#E57373',
    '🟡 Moderate':  '#FFD54F',
    '🟢 On Track':  '#81C784'
}
 
 
def plot_macro_prediction_chart(pred_df, strand, grade_level):
    """
    Horizontal bar chart of predicted cohort mean grades per subject.
    Bars are color-coded by risk level.
    White diamond markers show the most recent actual mean for reference.
    """
    if pred_df.empty:
        return go.Figure()
 
    p = pred_df.sort_values('predicted_mean', ascending=True).copy()
    bar_colors = p['risk_label'].map(_RISK_COLORS).fillna('#81C784').tolist()
 
    fig = go.Figure()
 
    # Predicted mean bars
    fig.add_trace(go.Bar(
        y=p['course'], x=p['predicted_mean'],
        orientation='h',
        marker_color=bar_colors,
        marker_line_width=0,
        name='Predicted Mean',
        customdata=np.stack((
            p['risk_label'],
            (p['risk_probability'] * 100).round(1).astype(str) + '%'
        ), axis=-1),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Predicted Mean: %{x:.2f}<br>'
            'Status: %{customdata[0]}<br>'
            'Risk Probability: %{customdata[1]}<extra></extra>'
        )
    ))
 
    # Prior actual mean reference dots
    if 'prior_mean' in p.columns:
        fig.add_trace(go.Scatter(
            y=p['course'], x=p['prior_mean'],
            mode='markers', name='Prior Actual Mean',
            marker=dict(size=9, color='white', symbol='diamond',
                        line=dict(color='grey', width=1.5)),
            hovertemplate='<b>%{y}</b><br>Prior Mean: %{x:.2f}<extra></extra>'
        ))
 
    # Threshold line at AT_RISK_THRESHOLD
    fig.add_vline(
        x=AT_RISK_THRESHOLD, line_dash='dot',
        line_color='rgba(229,115,115,0.6)', line_width=2,
        annotation_text=f'Risk Threshold ({AT_RISK_THRESHOLD})',
        annotation_position='top right',
        annotation_font_color='#E57373'
    )
 
    fig.update_layout(
        title=f'Predicted Subject Performance — {strand} Grade {grade_level}',
        xaxis_title='Predicted Mean Grade', xaxis=dict(range=[70, 100]),
        yaxis_title='Subject',
        legend=dict(orientation='h', y=-0.15),
        height=max(420, len(p) * 28 + 130),
        margin=dict(l=220, r=40, t=60, b=90)
    )
    return fig
 
 
def plot_micro_prediction_chart(pred_df, student_name):
    """
    Dumbbell-style chart showing actual grade (blue dot) vs. predicted grade
    (risk-colored dot) per subject. Grey connector shows the prediction delta.
    Sorted by risk probability descending.
    """
    if pred_df.empty:
        return go.Figure()
 
    p = pred_df.sort_values('risk_probability', ascending=False).copy()
    dot_colors = p['risk_label'].map(_RISK_COLORS).fillna('#81C784').tolist()
 
    fig = go.Figure()
 
    # Connector lines between actual and predicted
    for _, row in p.iterrows():
        if pd.notna(row.get('numeric_grade')):
            fig.add_trace(go.Scatter(
                x=[row['numeric_grade'], row['predicted_grade']],
                y=[row['course'],        row['course']],
                mode='lines',
                line=dict(color='rgba(180,180,180,0.45)', width=1.5),
                showlegend=False, hoverinfo='none'
            ))
 
    # Actual grade dots
    actual_mask = p['numeric_grade'].notna()
    if actual_mask.any():
        fig.add_trace(go.Scatter(
            x=p.loc[actual_mask, 'numeric_grade'],
            y=p.loc[actual_mask, 'course'],
            mode='markers', name='Actual Grade',
            marker=dict(size=10, color='#90CAF9',
                        line=dict(width=1.5, color='white')),
            hovertemplate='<b>%{y}</b><br>Actual Grade: %{x}<extra></extra>'
        ))
 
    # Predicted grade dots (risk-colored)
    fig.add_trace(go.Scatter(
        x=p['predicted_grade'], y=p['course'],
        mode='markers', name='Predicted Grade',
        marker=dict(size=12, color=dot_colors,
                    line=dict(width=1.5, color='white')),
        customdata=np.stack((
            p['risk_label'],
            (p['risk_probability'] * 100).round(1).astype(str) + '%'
        ), axis=-1),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Predicted: %{x:.2f}<br>'
            '%{customdata[0]} — %{customdata[1]}<extra></extra>'
        )
    ))
 
    # Peer mean reference dots
    if 'peer_mean' in p.columns and p['peer_mean'].notna().any():
        fig.add_trace(go.Scatter(
            x=p['peer_mean'], y=p['course'],
            mode='markers', name='Peer Mean',
            marker=dict(size=7, color='rgba(255,255,255,0.4)',
                        symbol='circle-open'),
            hovertemplate='<b>%{y}</b><br>Peer Mean: %{x:.2f}<extra></extra>'
        ))
 
    fig.add_vline(
        x=AT_RISK_THRESHOLD, line_dash='dot',
        line_color='rgba(229,115,115,0.6)', line_width=2
    )
 
    fig.update_layout(
        title=f'Predictive Grade Outlook — {student_name}',
        xaxis_title='Grade', xaxis=dict(range=[60, 100]),
        yaxis_title='Subject',
        legend=dict(orientation='h', y=-0.15),
        height=max(420, len(p) * 28 + 130),
        margin=dict(l=220, r=40, t=60, b=90)
    )
    return fig
 
 
def plot_feature_importance(fi_df, title='Feature Importance'):
    """
    Horizontal bar chart of Random Forest feature importances
    (Mean Decrease in Impurity). Human-readable feature labels.
    """
    if fi_df.empty:
        return go.Figure()
 
    fi = fi_df.copy()
    fi['Label'] = fi['Feature'].map(_FEATURE_LABELS).fillna(fi['Feature'])
    fi = fi.sort_values('Importance', ascending=True)
 
    fig = go.Figure(go.Bar(
        x=fi['Importance'], y=fi['Label'],
        orientation='h',
        marker_color='#7986CB',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))
 
    fig.update_layout(
        title=title,
        xaxis_title='Feature Importance (Mean Decrease in Impurity)',
        height=max(320, len(fi) * 30 + 110),
        margin=dict(l=200, r=30, t=55, b=55)
    )
    return fig