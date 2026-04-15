import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

# Page config for mobile responsiveness
st.set_page_config(page_title="Attendance Tracker", layout="centered")

st.title("📱 Class Attendance System")

# --- CONNECT TO GOOGLE SHEETS ---
# Uses the 'gsheets' connection defined in your Streamlit Secrets
conn = st.connection("gsheets", type=GSheetsConnection)

# --- SIDEBAR: SETUP & CONFIG ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload Student List (LMS CSV)", type="csv")
    attendance_date = st.date_input("Attendance Date", datetime.now())
    date_str = attendance_date.strftime("%Y-%m-%d")

if uploaded_file:
    # 1. Load the student list from your LMS export
    master_df = pd.read_csv(uploaded_file)
    
    # Check for specific LMS columns
    if 'Section' in master_df.columns and 'Student' in master_df.columns:
        sections = sorted(master_df['Section'].unique())
        selected_section = st.sidebar.selectbox("Select Section", sections)

        # Filter for the specific class section
        section_df = master_df[master_df['Section'] == selected_section].copy()

        # 2. Fetch data from Google Sheets to check for existing records
        try:
            # ttl=0 ensures we don't use cached data when marking attendance
            existing_data = conn.read(ttl=0)
        except:
            # Create empty dataframe if the sheet is brand new
            existing_data = pd.DataFrame(columns=['date', 'section', 'student_id', 'student_name', 'status_text', 'status_numeric'])

        # 3. Pre-fill statuses if you've already saved data for this section today
        current_day_records = existing_data[
            (existing_data['date'] == date_str) & 
            (existing_data['section'] == selected_section)
        ]

        if not current_day_records.empty:
            section_df = section_df.merge(
                current_day_records[['student_id', 'status_text']], 
                left_on='SIS User ID', right_on='student_id', how='left'
            )
            section_df['Attendance'] = section_df['status_text'].fillna("❌ Absent")
        else:
            section_df['Attendance'] = "❌ Absent"

        # 4. Mobile-Friendly Interface
        st.subheader(f"Section: {selected_section}")
        st.info(f"Marking for: {date_str}")

        edited_df = st.data_editor(
            section_df[['Attendance', 'Student', 'SIS User ID']],
            column_config={
                "Attendance": st.column_config.SelectboxColumn(
                    "Status",
                    options=["✅ Present", "❌ Absent"], #
                    width="medium",
                    required=True,
                ),
                "Student": st.column_config.TextColumn("Student Name", disabled=True),
                "SIS User ID": None # Hidden to save screen space on mobile
            },
            hide_index=True,
            use_container_width=True
        )

        # 5. Syncing Logic
        if st.button("💾 Sync to Google Sheets", use_container_width=True, type="primary"):
            new_entries = []
            for _, row in edited_df.iterrows():
                new_entries.append({
                    'date': date_str,
                    'section': selected_section,
                    'student_id': row['SIS User ID'],
                    'student_name': row['Student'],
                    'status_text': row['Attendance'],
                    'status_numeric': 1 if row['Attendance'] == "✅ Present" else 0 #
                })
            
            new_df = pd.DataFrame(new_entries)

            # Combine old data with new marks and remove duplicates
            updated_db = pd.concat([existing_data, new_df])
            updated_db = updated_db.drop_duplicates(subset=['student_id', 'date'], keep='last')

            conn.update(data=updated_db)
            st.success("Attendance synced to Cloud!")
    else:
        st.error("CSV must contain 'Section', 'Student', and 'SIS User ID' columns.")

# --- EXPORT SECTION ---
st.divider()
if st.button("📥 Download Master Numerical Report", use_container_width=True):
    final_db = conn.read(ttl=0)
    if not final_db.empty:
        # Exporting numeric values for easy computation
        csv = final_db.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"Master_Attendance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            use_container_width=True
        )