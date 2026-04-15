# ==============================
# GLOBAL CONSTANTS
# ==============================

GRADE_MIN = 60
GRADE_MAX = 100

PASSING_GRADE = 75
AT_RISK_THRESHOLD = 80

TOP_QUANTILE = 0.8
BOTTOM_QUANTILE = 0.2

RECENT_YEARS_WINDOW = 3

# ==============================
# COLUMN NAMES (Canvas LMS)
# ==============================

COL_STUDENT = "student sis"
COL_COURSE = "course"
COL_SECTION = "section sis"
COL_SCORE = "final score"

# ==============================
# SUBJECT NAME STANDARDIZATION
# ==============================

SUBJECT_MAPPING = {
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
    'Pagbasa at Pagsusuri ng Ibat Ibang Teksto Tungo sa Pananaliksik': 'Pagbasa at Pagsusuri ng Iba\'t Ibang Teksto Tungo sa Pananaliksik',
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

def process_section_info(row):
    """
    Purpose:
        Extract strand and grade level from section identifier.

    Parameters:
        row (pd.Series): Data row containing section information

    Returns:
        pd.Series: [strand, grade_level]

    Notes:
        - Assumes section naming follows institutional conventions
    """
    section = str(row.get(COL_SECTION, "")).upper()

    # --- Strand Detection ---
    strand = next(
        (s for s in ["STEM", "ABM", "HUMSS", "GAS"] if s in section),
        "UNKNOWN"
    )

    # --- Grade Level Detection ---
    match = re.search(r"(11|12)", section)
    grade_level = match.group(0) if match else "UNKNOWN"

    return pd.Series([strand, grade_level])
    
def clean_subject_names(df):
    """
    Purpose:
        Standardize subject/course names.

    Parameters:
        df (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Dataset with cleaned subject names

    Notes:
        - Preserves mapping dictionary logic (DO NOT REMOVE)
    """
    df[COL_COURSE] = df[COL_COURSE].str.strip().str.upper()
    df[COL_COURSE] = df[COL_COURSE].replace(SUBJECT_MAPPING)
    return df

def load_and_process_data(csv_files):
    """
    Purpose:
        Load and preprocess Canvas LMS grade exports.

    Parameters:
        csv_files (list[str]): List of CSV file paths

    Returns:
        pd.DataFrame: Cleaned and feature-engineered dataset

    Notes:
        - Assumes standardized institutional schema
        - Drops rows with invalid numeric grades
    """
    frames = []

    # ==============================
    # LOAD FILES
    # ==============================
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = file
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # ==============================
    # CLEANING
    # ==============================
    df = clean_subject_names(df)

    df[["strand", "grade_level"]] = df.apply(
        process_section_info,
        axis=1,
        result_type="expand"
    )

    # ==============================
    # FEATURE ENGINEERING
    # ==============================
    df["numeric_grade"] = pd.to_numeric(df[COL_SCORE], errors="coerce")

    df["school_year"] = df["source_file"].str.extract(r"(\d{4}-\d{4})")
    df["semester"] = df["source_file"].str.extract(r"_(\d)")
    df["full_term"] = df["school_year"] + " S" + df["semester"]

    # ==============================
    # FINAL FILTER
    # ==============================
    df = df.dropna(subset=["numeric_grade"])

    return df

def get_overview_metrics(df):
    """
    Purpose:
        Compute high-level performance metrics.

    Parameters:
        df (pd.DataFrame): Clean dataset

    Returns:
        dict: Overview statistics

    Notes:
        - Includes strand-level averages and pass rates
    """
    if df.empty:
        return None

    result = {
        "Total Students": df[COL_STUDENT].nunique(),
        "Strands": {}
    }

    for strand, group in df.groupby("strand"):
        result["Strands"][strand] = {
            "Average Grade": group["numeric_grade"].mean(),
            "Passing Rate (%)": (
                (group["numeric_grade"] >= PASSING_GRADE).mean() * 100
            )
        }

    return result

def get_subgroup_statistics(df, year, grade_level, strand, mode="top"):
    """
    Purpose:
        Compute statistics for top or bottom student subgroups.

    Parameters:
        df (pd.DataFrame): Dataset
        year (str): School year
        grade_level (str): Grade level
        strand (str): Academic strand
        mode (str): "top" or "bottom"

    Returns:
        dict: Subgroup statistics

    Notes:
        - Uses quantile-based filtering
        - Preserves original cohort selection logic
    """
    subset = df[
        (df["school_year"] == year) &
        (df["grade_level"] == grade_level) &
        (df["strand"] == strand)
    ]

    if subset.empty:
        return None

    student_avg = subset.groupby(COL_STUDENT)["numeric_grade"].mean()

    if mode == "top":
        selected_ids = student_avg[student_avg >= student_avg.quantile(TOP_QUANTILE)].index
    else:
        selected_ids = student_avg[student_avg <= student_avg.quantile(BOTTOM_QUANTILE)].index

    subgroup = subset[subset[COL_STUDENT].isin(selected_ids)]

    return {
        "num_students": len(selected_ids),
        "avg_grade": subgroup["numeric_grade"].mean(),
        "subject_avg": subgroup.groupby(COL_COURSE)["numeric_grade"].mean().to_dict()
    }

def plot_grade_distribution_interactive(df):
    """
    Purpose:
        Visualize grade distribution across school years and strands.

    Returns:
        Plotly Figure
    """
    fig = px.box(
        df,
        x="school_year",
        y="numeric_grade",
        color="strand"
    )

    fig = apply_standard_layout(fig, "Grade Distribution by Year and Strand")
    return apply_grade_axis(fig)

def plot_grade_density_interactive(df):
    """
    Purpose:
        Show grade density distribution across school years.

    Returns:
        Plotly Figure
    """
    fig = px.histogram(
        df,
        x="numeric_grade",
        color="school_year",
        nbins=30,
        histnorm="density",
        opacity=0.6
    )

    fig = apply_standard_layout(fig, "Grade Density Distribution")
    return fig

def plot_subject_extremes_split(df):
    """
    Purpose:
        Identify and visualize easiest and hardest subjects per strand.

    Returns:
        tuple: (hardest_fig, easiest_fig)
    """
    subject_avg = df.groupby(["strand", COL_COURSE])["numeric_grade"].mean().reset_index()

    hardest = subject_avg.sort_values("numeric_grade").groupby("strand").head(5)
    easiest = subject_avg.sort_values("numeric_grade", ascending=False).groupby("strand").head(5)

    fig_hard = px.bar(
        hardest,
        x="numeric_grade",
        y=COL_COURSE,
        color="strand",
        orientation="h"
    )

    fig_easy = px.bar(
        easiest,
        x="numeric_grade",
        y=COL_COURSE,
        color="strand",
        orientation="h"
    )

    fig_hard = apply_standard_layout(fig_hard, "Hardest Subjects")
    fig_easy = apply_standard_layout(fig_easy, "Easiest Subjects")

    return fig_hard, fig_easy

def plot_subject_deep_dive_interactive(df, subject):
    """
    Purpose:
        Analyze distribution of a single subject across years.

    Parameters:
        subject (str): Subject name

    Returns:
        Plotly Figure
    """
    subset = df[df[COL_COURSE] == subject]

    if subset.empty:
        return None

    fig = px.histogram(
        subset,
        x="numeric_grade",
        color="school_year",
        nbins=25,
        opacity=0.6
    )

    fig = apply_standard_layout(fig, f"{subject} Performance Distribution")
    return apply_grade_axis(fig)

def plot_subject_comparison_interactive(df, subject1, subject2):
    """
    Purpose:
        Compare distributions of two subjects.

    Returns:
        Plotly Figure
    """
    subset = df[df[COL_COURSE].isin([subject1, subject2])]

    fig = px.box(
        subset,
        x=COL_COURSE,
        y="numeric_grade",
        color=COL_COURSE
    )

    fig = apply_standard_layout(fig, f"{subject1} vs {subject2}")
    return apply_grade_axis(fig)

def plot_correlation_heatmap_interactive(df, year, grade, strand):
    """
    Purpose:
        Generate subject correlation heatmap.

    Returns:
        (fig, error_message)
    """
    subset = df[
        (df["school_year"] == year) &
        (df["grade_level"] == grade) &
        (df["strand"] == strand)
    ]

    if subset.empty:
        return None, "No data available for selected filters."

    pivot = subset.pivot_table(
        index=COL_STUDENT,
        columns=COL_COURSE,
        values="numeric_grade"
    )

    corr = pivot.corr()

    fig = px.imshow(corr, text_auto=True)

    fig = apply_standard_layout(fig, "Subject Correlation Heatmap")
    return fig, None

def build_macro_features(df):
    """
    Purpose:
        Construct cohort-level features for macro modeling.

    Returns:
        pd.DataFrame
    """
    grouped = df.groupby(
        ["school_year", "strand", "grade_level"]
    )["numeric_grade"].mean().reset_index()

    # Create target
    grouped["target"] = grouped["numeric_grade"]

    # Train/Test split (temporal logic preserved)
    latest_year = grouped["school_year"].max()
    grouped["is_train"] = grouped["school_year"] != latest_year

    return grouped

def train_macro_model(df):
    """
    Purpose:
        Train cohort-level regression and classification models.

    Returns:
        tuple:
            (reg_model, clf_model, metrics dict, feature importance df)
    """
    features = build_macro_features(df)

    if features.empty:
        return None, None, {"error": "Insufficient data"}, pd.DataFrame()

    train_df = features[features["is_train"]]
    test_df = features[~features["is_train"]]

    # ==============================
    # FEATURE PREP
    # ==============================
    X_train = pd.get_dummies(train_df[["strand", "grade_level"]])
    X_test = pd.get_dummies(test_df[["strand", "grade_level"]])

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df["target"]
    y_test = test_df["target"]

    # ==============================
    # REGRESSION MODEL
    # ==============================
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    reg_preds = reg.predict(X_test)

    # ==============================
    # CLASSIFICATION MODEL
    # ==============================
    y_train_clf = (y_train < AT_RISK_THRESHOLD).astype(int)
    y_test_clf = (y_test < AT_RISK_THRESHOLD).astype(int)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train_clf)

    clf_probs = clf.predict_proba(X_test)[:, 1]

    # ==============================
    # METRICS
    # ==============================
    metrics = {
        "MAE": mean_absolute_error(y_test, reg_preds),
        "R2": r2_score(y_test, reg_preds),
        "ROC_AUC": roc_auc_score(y_test_clf, clf_probs),
        "train_size": len(train_df),
        "test_size": len(test_df)
    }

    # ==============================
    # FEATURE IMPORTANCE
    # ==============================
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": reg.feature_importances_
    }).sort_values("importance", ascending=False)

    return reg, clf, metrics, fi

def build_micro_features(df):
    """
    Purpose:
        Construct student-level features.

    Returns:
        pd.DataFrame
    """
    df = df.copy()

    df["student_avg"] = df.groupby(COL_STUDENT)["numeric_grade"].transform("mean")

    df["target"] = df["numeric_grade"]

    latest_term = df["full_term"].max()
    df["is_train"] = df["full_term"] != latest_term

    return df


def train_micro_model(df):
    """
    Purpose:
        Train student-level regression and classification models.

    Returns:
        tuple:
            (reg_model, clf_model, metrics dict, feature importance df)
    """
    features = build_micro_features(df)

    if features.empty:
        return None, None, {"error": "Insufficient data"}, pd.DataFrame()

    train_df = features[features["is_train"]]
    test_df = features[~features["is_train"]]

    X_train = train_df[["student_avg"]]
    X_test = test_df[["student_avg"]]

    y_train = train_df["target"]
    y_test = test_df["target"]

    # ==============================
    # REGRESSION
    # ==============================
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    reg_preds = reg.predict(X_test)

    # ==============================
    # CLASSIFICATION
    # ==============================
    y_train_clf = (y_train < AT_RISK_THRESHOLD).astype(int)
    y_test_clf = (y_test < AT_RISK_THRESHOLD).astype(int)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train_clf)

    clf_probs = clf.predict_proba(X_test)[:, 1]

    # ==============================
    # METRICS
    # ==============================
    metrics = {
        "MAE": mean_absolute_error(y_test, reg_preds),
        "R2": r2_score(y_test, reg_preds),
        "ROC_AUC": roc_auc_score(y_test_clf, clf_probs),
        "train_size": len(train_df),
        "test_size": len(test_df)
    }

    # ==============================
    # FEATURE IMPORTANCE
    # ==============================
    fi = pd.DataFrame({
        "feature": ["student_avg"],
        "importance": reg.feature_importances_
    })

    return reg, clf, metrics, fi

def predict_macro_outlook(model_reg, model_clf, strand, grade_level):
    """
    Purpose:
        Predict cohort performance and risk.

    Returns:
        dict
    """
    X = pd.DataFrame([{
        "strand": strand,
        "grade_level": grade_level
    }])

    X = pd.get_dummies(X)

    # Align columns if needed handled externally

    pred_grade = model_reg.predict(X)[0]
    prob_risk = model_clf.predict_proba(X)[0][1]

    return {
        "predicted_grade": pred_grade,
        "risk_probability": prob_risk,
        "risk_label": get_risk_label(prob_risk)
    }


def get_student_profile(df, student_id):
    """
    Purpose:
        Retrieve complete academic profile of a student.

    Parameters:
        student_id (str): Student identifier

    Returns:
        pd.DataFrame
    """
    student_df = df[df[COL_STUDENT] == student_id]

    if student_df.empty:
        return None

    return student_df.sort_values("full_term")

def plot_student_performance_trend(df, student_id):
    """
    Purpose:
        Visualize student's grade trend over time.

    Returns:
        Plotly Figure
    """
    student_df = get_student_profile(df, student_id)

    if student_df is None:
        return None

    trend = student_df.groupby("full_term")["numeric_grade"].mean().reset_index()

    fig = px.line(
        trend,
        x="full_term",
        y="numeric_grade",
        markers=True
    )

    fig = apply_standard_layout(fig, "Student Performance Trend")
    return apply_grade_axis(fig)

def compare_student_to_cohort(df, student_id):
    """
    Purpose:
        Compare student performance against cohort.

    Returns:
        dict
    """
    student_df = get_student_profile(df, student_id)

    if student_df is None:
        return None

    latest = student_df.iloc[-1]

    cohort = df[
        (df["strand"] == latest["strand"]) &
        (df["grade_level"] == latest["grade_level"]) &
        (df["full_term"] == latest["full_term"])
    ]

    student_avg = student_df["numeric_grade"].mean()
    cohort_avg = cohort["numeric_grade"].mean()

    return {
        "student_avg": student_avg,
        "cohort_avg": cohort_avg,
        "difference": student_avg - cohort_avg
    }

def get_student_subject_strengths(df, student_id):
    """
    Purpose:
        Identify strongest and weakest subjects.

    Returns:
        dict
    """
    student_df = get_student_profile(df, student_id)

    if student_df is None:
        return None

    subject_avg = student_df.groupby(COL_COURSE)["numeric_grade"].mean()

    strongest = subject_avg.sort_values(ascending=False).head(3)
    weakest = subject_avg.sort_values().head(3)

    return {
        "strongest": strongest.to_dict(),
        "weakest": weakest.to_dict()
    }

def predict_student_performance(model_reg, model_clf, df, student_id):
    """
    Purpose:
        Predict student's future performance and risk.

    Returns:
        dict
    """
    student_df = get_student_profile(df, student_id)

    if student_df is None:
        return None

    student_avg = student_df["numeric_grade"].mean()

    X = pd.DataFrame([{"student_avg": student_avg}])

    predicted_grade = model_reg.predict(X)[0]
    risk_prob = model_clf.predict_proba(X)[0][1]

    return {
        "predicted_grade": predicted_grade,
        "risk_probability": risk_prob,
        "risk_label": get_risk_label(risk_prob)
    }

def extract_student_curriculum(df, student_id):
    """
    Purpose:
        Extract subjects taken by student across timeline.

    Returns:
        dict
    """
    student_df = get_student_profile(df, student_id)

    if student_df is None:
        return None

    curriculum = (
        student_df
        .groupby("full_term")[COL_COURSE]
        .apply(list)
        .to_dict()
    )

    return curriculum

def get_student_percentile(df, student_id):
    """
    Purpose:
        Determine student's percentile rank within cohort.

    Returns:
        float
    """
    student_df = get_student_profile(df, student_id)

    if student_df is None:
        return None

    latest = student_df.iloc[-1]

    cohort = df[
        (df["strand"] == latest["strand"]) &
        (df["grade_level"] == latest["grade_level"]) &
        (df["full_term"] == latest["full_term"])
    ]

    student_avg = student_df["numeric_grade"].mean()
    cohort_avgs = cohort.groupby(COL_STUDENT)["numeric_grade"].mean()

    percentile = (cohort_avgs < student_avg).mean() * 100

    return percentile


