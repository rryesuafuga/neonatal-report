"""
Mbale Regional Referral Hospital — Neonatal Unit Monthly Reporting Tool
========================================================================
Developed for Dr Adam Hewitt-Smith & Dr Kathy Hewitt-Smith
By Raymond Reuel Wayesu, Biostatistician

This Streamlit application automates the generation of monthly neonatal
reports from the Mbale RRH Neonatal Unit logbook data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import warnings
import os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION & STYLING
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mbale RRH Neonatal Reporting Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lancet-inspired colour palette
LANCET_COLORS = [
    "#00468B",  # Dark blue
    "#ED0000",  # Red
    "#42B540",  # Green
    "#0099B4",  # Teal
    "#925E9F",  # Purple
    "#FDAF91",  # Peach
    "#AD002A",  # Dark red
    "#ADB6B6",  # Grey
    "#1B1919",  # Near-black
    "#E0A800",  # Amber
]

LANCET_BG = "#FFFFFF"
LANCET_GRID = "#E5E5E5"
LANCET_TEXT = "#1B1919"
LANCET_FONT = "Arial, Helvetica, sans-serif"

# Inject custom CSS for Lancet-style tables
st.markdown("""
<style>
    /* Overall app styling */
    .main .block-container { max-width: 1200px; padding-top: 1.5rem; }
    
    /* Lancet-style table */
    .lancet-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 13px;
        color: #1B1919;
        margin: 12px 0;
    }
    .lancet-table thead th {
        background-color: #00468B;
        color: #FFFFFF;
        font-weight: 600;
        padding: 8px 12px;
        text-align: left;
        border-bottom: 2px solid #00468B;
        font-size: 13px;
    }
    .lancet-table tbody td {
        padding: 6px 12px;
        border-bottom: 1px solid #E5E5E5;
        font-size: 13px;
    }
    .lancet-table tbody tr:nth-child(even) {
        background-color: #F8F9FA;
    }
    .lancet-table tbody tr:hover {
        background-color: #EDF2F7;
    }
    .lancet-table tfoot td {
        font-weight: 700;
        border-top: 2px solid #00468B;
        padding: 8px 12px;
        background-color: #F0F4F8;
        font-size: 13px;
    }
    
    /* Section headers */
    .report-section {
        font-family: Arial, Helvetica, sans-serif;
        color: #00468B;
        border-bottom: 2px solid #00468B;
        padding-bottom: 4px;
        margin-top: 24px;
        margin-bottom: 12px;
    }
    
    /* Metric cards */
    .metric-card {
        background: #F8F9FA;
        border-left: 4px solid #00468B;
        padding: 12px 16px;
        border-radius: 0 4px 4px 0;
        margin: 6px 0;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #00468B;
    }
    .metric-label {
        font-size: 13px;
        color: #666;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F0F4F8;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# DATA PROCESSING FUNCTIONS
# ──────────────────────────────────────────────────────────────

def classify_facility_level(name):
    """Classify health facility by Uganda health system level."""
    if pd.isna(name) or str(name).strip() == "":
        return "Unknown"
    name_upper = str(name).strip().upper()
    if name_upper in ("HOME",):
        return "Home"
    if "BBA" in name_upper:
        return "Born Before Arrival (BBA)"
    if "RRH" in name_upper:
        return "Regional Referral Hospital"
    if "G.H" in name_upper or "GENERAL HOSPITAL" in name_upper:
        return "General Hospital"
    if "D.H" in name_upper or "DISTRICT HOSPITAL" in name_upper:
        return "District Hospital"
    if "HOSPITAL" in name_upper:
        return "Hospital (Other)"
    if "HCIV" in name_upper:
        return "Health Centre IV"
    if "HCIII" in name_upper:
        return "Health Centre III"
    if "HCII" in name_upper:
        return "Health Centre II"
    # Catch private clinics and others
    return "Other (Private Clinic/Not Classified)"


def categorize_birthweight(bw_kg):
    """Categorize birth weight per WHO neonatal categories."""
    if pd.isna(bw_kg):
        return "Unknown"
    if bw_kg < 1.0:
        return "<1000g"
    elif bw_kg < 1.5:
        return "1000–1499g"
    elif bw_kg < 2.5:
        return "1500–2499g"
    else:
        return "≥2500g"


def map_outcome(code):
    """Map outcome codes to readable labels."""
    if pd.isna(code) or str(code).strip() == "":
        return "Unknown"
    code = str(code).strip().upper()
    mapping = {
        "D": "Discharged",
        "DD": "Died",
        "R": "Referred Out",
        "S": "Self-Discharge / Runaway",
        "T": "Transferred",
        "RAB": "Re-Admitted / Brought Back",
    }
    return mapping.get(code, f"Other ({code})")


def derive_combined_outcome(row):
    """
    Combine 7-day and 28-day outcomes into a single hospital outcome.
    Priority: If either shows death, outcome is death.
    Otherwise, use the latest available outcome.
    """
    s7 = str(row.get("Status at 7 Days - Outcome", "")).strip().upper() if pd.notna(row.get("Status at 7 Days - Outcome")) else ""
    s28 = str(row.get("Status at 28 Days - Outcome", "")).strip().upper() if pd.notna(row.get("Status at 28 Days - Outcome")) else ""
    
    if s7 == "DD" or s28 == "DD":
        return "Died"
    if s28 != "":
        return map_outcome(s28)
    if s7 != "":
        return map_outcome(s7)
    return "Unknown"


def clean_data(df):
    """Master data cleaning and derivation pipeline."""
    df = df.copy()
    
    # Drop fully empty trailing column if present
    if "Unnamed: 28" in df.columns:
        df.drop(columns=["Unnamed: 28"], inplace=True)
    
    # ── Parse dates ──
    df["Date of Admission"] = pd.to_datetime(
        df["Date of Admission"], format="%d.%m.%y", errors="coerce"
    )
    df["Date of Birth"] = pd.to_datetime(
        df["Date of Birth"], format="%d.%m.%y", errors="coerce"
    )
    df["Admission_Month"] = df["Date of Admission"].dt.to_period("M")
    df["Admission_Month_Label"] = df["Date of Admission"].dt.strftime("%b %Y")
    
    # ── Standardize categorical fields ──
    # HIV Status
    df["HIV Status Code"] = df["HIV Status Code"].fillna("Unknown").str.strip()
    df["HIV Status Code"] = df["HIV Status Code"].replace({"unknown": "Unknown", "": "Unknown"})
    
    # Mode of Delivery
    df["Mode of Delivery"] = df["Mode of Delivery"].fillna("Unknown").str.strip()
    # Fix case inconsistencies
    mode_map = {
        "spontaneous Vaginal Delivery": "Spontaneous Vaginal Delivery",
        "caesarean Section": "Caesarean Section",
        "": "Unknown",
    }
    df["Mode of Delivery"] = df["Mode of Delivery"].replace(mode_map)
    
    # Sex
    df["Sex"] = df["Sex"].fillna("Unknown").str.strip()
    df["Sex"] = df["Sex"].replace({"Amoprodile": "Ambiguous", "": "Unknown"})
    
    # Multiple births
    df["How many babies born"] = df["How many babies born"].fillna("Unknown").str.strip()
    df["How many babies born"] = df["How many babies born"].replace({
        "singleton": "Singleton", "": "Unknown"
    })
    
    # Final Diagnosis
    df["Final Diagnosis"] = df["Final Diagnosis"].fillna("Unknown").str.strip()
    # Standardize common case issues
    diagnosis_map = {
        "bacterial Sepsis of newborn, unspecified": "Bacterial Sepsis of newborn, unspecified",
        "neonatal Jaundice, unspecified": "Neonatal Jaundice, unspecified",
        "": "Unknown",
    }
    df["Final Diagnosis"] = df["Final Diagnosis"].replace(diagnosis_map)
    
    # ── Derived fields ──
    df["Birth Weight (Kg)"] = pd.to_numeric(df["Birth Weight (Kg)"], errors="coerce")
    df["BW_Category"] = df["Birth Weight (Kg)"].apply(categorize_birthweight)
    # Ordered category for proper sorting
    bw_order = ["<1000g", "1000–1499g", "1500–2499g", "≥2500g", "Unknown"]
    df["BW_Category"] = pd.Categorical(df["BW_Category"], categories=bw_order, ordered=True)
    
    df["Place_of_Birth_Level"] = df["Place of Birth"].apply(classify_facility_level)
    df["Referral_From_Level"] = df["Referral From"].apply(classify_facility_level)
    
    df["Combined_Outcome"] = df.apply(derive_combined_outcome, axis=1)
    df["Status_7d"] = df["Status at 7 Days - Outcome"].apply(map_outcome)
    df["Status_28d"] = df["Status at 28 Days - Outcome"].apply(map_outcome)
    df["Died"] = df["Combined_Outcome"] == "Died"
    
    return df


# ──────────────────────────────────────────────────────────────
# TABLE & FIGURE RENDERING FUNCTIONS
# ──────────────────────────────────────────────────────────────

def render_lancet_table(df_table, title="", footer_row=None, table_num=None):
    """Render a DataFrame as a Lancet-style HTML table."""
    table_label = f"<strong>Table {table_num}.</strong> " if table_num else ""
    
    html = f'<p style="font-family: {LANCET_FONT}; font-size: 13px; color: {LANCET_TEXT}; margin-bottom: 4px;">'
    html += f'{table_label}<em>{title}</em></p>'
    html += '<table class="lancet-table"><thead><tr>'
    
    for col in df_table.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    for _, row in df_table.iterrows():
        html += "<tr>"
        for val in row:
            html += f"<td>{val}</td>"
        html += "</tr>"
    
    if footer_row is not None:
        html += "</tbody><tfoot><tr>"
        for val in footer_row:
            html += f"<td>{val}</td>"
        html += "</tr></tfoot>"
    else:
        html += "</tbody>"
    
    html += "</table>"
    return html


def lancet_plotly_layout(fig, title="", xaxis_title="", yaxis_title="", height=450):
    """Apply Lancet journal styling to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=LANCET_FONT, size=14, color=LANCET_TEXT),
            x=0.0,
            xanchor="left",
        ),
        font=dict(family=LANCET_FONT, size=12, color=LANCET_TEXT),
        plot_bgcolor=LANCET_BG,
        paper_bgcolor=LANCET_BG,
        xaxis=dict(
            title=xaxis_title,
            showgrid=False,
            linecolor=LANCET_TEXT,
            linewidth=1,
            ticks="outside",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=LANCET_GRID,
            gridwidth=0.5,
            linecolor=LANCET_TEXT,
            linewidth=1,
            ticks="outside",
            tickfont=dict(size=11),
        ),
        legend=dict(
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=LANCET_GRID,
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=50, b=60),
        height=height,
    )
    return fig


def freq_table(series, name="Category", total_label="Total"):
    """Create a frequency table with n and %."""
    counts = series.value_counts(dropna=False).sort_values(ascending=False)
    total = counts.sum()
    df_out = pd.DataFrame({
        name: counts.index.astype(str),
        "n": counts.values,
        "% of Total": [f"{(v / total * 100):.1f}%" for v in counts.values],
    })
    footer = [total_label, str(total), "100.0%"]
    return df_out, footer


# ──────────────────────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────────────────────

def generate_report_section(df, section_prefix="", table_counter_start=1):
    """Generate all 10 report sections for a given filtered dataframe."""
    tc = table_counter_start  # table counter
    tables_for_export = {}
    figures_for_export = {}

    n_total = len(df)

    if n_total == 0:
        st.warning("No records match the current filter criteria.")
        return tables_for_export, tc, figures_for_export
    
    # ── 1. Total admissions per month ──
    st.markdown(f'<h4 class="report-section">{section_prefix}1. Total Admissions per Month</h4>', unsafe_allow_html=True)
    
    monthly = df.groupby("Admission_Month").size().reset_index(name="Admissions")
    monthly = monthly.sort_values("Admission_Month")
    monthly["Month"] = monthly["Admission_Month"].dt.strftime("%b %Y")
    total_admissions = monthly["Admissions"].sum()

    tbl = monthly[["Month", "Admissions"]].copy()
    tbl["% of Total"] = (tbl["Admissions"] / total_admissions * 100).round(1).astype(str) + "%"
    
    st.markdown(render_lancet_table(
        tbl, title=f"{section_prefix}Total neonatal admissions by month of admission",
        footer_row=["Total", str(total_admissions), "100.0%"],
        table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Admissions_Month"] = tbl
    tc += 1
    
    # Visualization — bar chart
    fig = go.Figure(go.Bar(
        x=monthly["Month"].astype(str),
        y=monthly["Admissions"],
        marker_color=LANCET_COLORS[0],
        text=monthly["Admissions"],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig = lancet_plotly_layout(
        fig,
        title=f"Figure. {section_prefix}Monthly neonatal admissions",
        xaxis_title="Month",
        yaxis_title="Number of Admissions",
    )
    st.plotly_chart(fig, use_container_width=True)
    figures_for_export[f"{section_prefix}Monthly_Admissions"] = fig

    # ── 2. Place of Birth ──
    st.markdown(f'<h4 class="report-section">{section_prefix}2. Place of Birth</h4>', unsafe_allow_html=True)
    
    # 2a. Summary by facility level
    pob_level, pob_level_footer = freq_table(df["Place_of_Birth_Level"], "Facility Level")
    st.markdown(render_lancet_table(
        pob_level, title=f"{section_prefix}Place of birth by facility level",
        footer_row=pob_level_footer, table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Birth_Level"] = pob_level
    tc += 1
    
    # 2b. Top 10 places of birth
    top10_pob = df["Place of Birth"].fillna("Unknown").value_counts().head(10).reset_index()
    top10_pob.columns = ["Place of Birth", "n"]
    top10_pob["% of Total"] = (top10_pob["n"] / n_total * 100).round(1).astype(str) + "%"
    st.markdown(render_lancet_table(
        top10_pob, title=f"{section_prefix}Top 10 places of birth",
        table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Top10_Birth"] = top10_pob
    tc += 1
    
    # Visualization — horizontal bar for facility level
    pob_viz = pob_level.copy()
    pob_viz["n"] = pob_viz["n"].astype(int)
    pob_viz = pob_viz.sort_values("n", ascending=True)
    fig2 = go.Figure(go.Bar(
        y=pob_viz["Facility Level"],
        x=pob_viz["n"],
        orientation="h",
        marker_color=LANCET_COLORS[0],
        text=pob_viz["n"],
        textposition="outside",
    ))
    fig2 = lancet_plotly_layout(
        fig2,
        title=f"Figure. {section_prefix}Place of birth by facility level",
        xaxis_title="Number of Admissions",
        yaxis_title="",
    )
    st.plotly_chart(fig2, use_container_width=True)
    figures_for_export[f"{section_prefix}Place_of_Birth"] = fig2

    # ── 3. Place of Referral ──
    st.markdown(f'<h4 class="report-section">{section_prefix}3. Place of Referral</h4>', unsafe_allow_html=True)
    
    ref_level, ref_level_footer = freq_table(df["Referral_From_Level"], "Facility Level")
    st.markdown(render_lancet_table(
        ref_level, title=f"{section_prefix}Referral source by facility level",
        footer_row=ref_level_footer, table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Referral_Level"] = ref_level
    tc += 1
    
    top10_ref = df["Referral From"].fillna("Unknown").value_counts().head(10).reset_index()
    top10_ref.columns = ["Referral From", "n"]
    top10_ref["% of Total"] = (top10_ref["n"] / n_total * 100).round(1).astype(str) + "%"
    st.markdown(render_lancet_table(
        top10_ref, title=f"{section_prefix}Top 10 referral sources",
        table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Top10_Referral"] = top10_ref
    tc += 1
    
    # ── 4. HIV Status ──
    st.markdown(f'<h4 class="report-section">{section_prefix}4. HIV Status</h4>', unsafe_allow_html=True)
    
    hiv_tbl, hiv_footer = freq_table(df["HIV Status Code"], "HIV Status")
    st.markdown(render_lancet_table(
        hiv_tbl, title=f"{section_prefix}Maternal HIV status of admitted neonates",
        footer_row=hiv_footer, table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_HIV"] = hiv_tbl
    tc += 1
    
    # Visualization — pie chart
    fig_hiv = go.Figure(go.Pie(
        labels=hiv_tbl["HIV Status"],
        values=hiv_tbl["n"],
        marker=dict(colors=LANCET_COLORS[:len(hiv_tbl)]),
        textinfo="label+percent",
        textfont=dict(size=12, family=LANCET_FONT),
        hole=0.3,
    ))
    fig_hiv = lancet_plotly_layout(fig_hiv, title=f"Figure. {section_prefix}Maternal HIV status", height=380)
    st.plotly_chart(fig_hiv, use_container_width=True)
    figures_for_export[f"{section_prefix}HIV_Status"] = fig_hiv

    # ── 5. Mode of Delivery ──
    st.markdown(f'<h4 class="report-section">{section_prefix}5. Mode of Delivery</h4>', unsafe_allow_html=True)
    
    mod_tbl, mod_footer = freq_table(df["Mode of Delivery"], "Mode of Delivery")
    st.markdown(render_lancet_table(
        mod_tbl, title=f"{section_prefix}Mode of delivery",
        footer_row=mod_footer, table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Delivery"] = mod_tbl
    tc += 1
    
    # ── 6. Sex ──
    st.markdown(f'<h4 class="report-section">{section_prefix}6. Sex</h4>', unsafe_allow_html=True)
    
    sex_tbl, sex_footer = freq_table(df["Sex"], "Sex")
    st.markdown(render_lancet_table(
        sex_tbl, title=f"{section_prefix}Sex distribution of admitted neonates",
        footer_row=sex_footer, table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Sex"] = sex_tbl
    tc += 1
    
    # Visualization — side by side sex + delivery
    col1, col2 = st.columns(2)
    with col1:
        fig_sex = go.Figure(go.Pie(
            labels=sex_tbl["Sex"], values=sex_tbl["n"],
            marker=dict(colors=[LANCET_COLORS[0], LANCET_COLORS[1], LANCET_COLORS[7]]),
            textinfo="label+percent", hole=0.3,
        ))
        fig_sex = lancet_plotly_layout(fig_sex, title=f"Figure. {section_prefix}Sex distribution", height=350)
        st.plotly_chart(fig_sex, use_container_width=True)
        figures_for_export[f"{section_prefix}Sex_Distribution"] = fig_sex
    with col2:
        fig_mod = go.Figure(go.Pie(
            labels=mod_tbl["Mode of Delivery"], values=mod_tbl["n"],
            marker=dict(colors=LANCET_COLORS[:len(mod_tbl)]),
            textinfo="label+percent", hole=0.3,
        ))
        fig_mod = lancet_plotly_layout(fig_mod, title=f"Figure. {section_prefix}Mode of delivery", height=350)
        st.plotly_chart(fig_mod, use_container_width=True)
        figures_for_export[f"{section_prefix}Mode_of_Delivery"] = fig_mod

    # ── 7. Multiple Births ──
    st.markdown(f'<h4 class="report-section">{section_prefix}7. Multiple Births</h4>', unsafe_allow_html=True)
    
    mult_tbl, mult_footer = freq_table(df["How many babies born"], "Birth Type")
    st.markdown(render_lancet_table(
        mult_tbl, title=f"{section_prefix}Singleton vs. multiple births",
        footer_row=mult_footer, table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_MultipleBirths"] = mult_tbl
    tc += 1
    
    # ── 8. Birthweight Categories ──
    st.markdown(f'<h4 class="report-section">{section_prefix}8. Birth Weight Categories</h4>', unsafe_allow_html=True)
    
    bw_counts = df["BW_Category"].value_counts().reindex(
        ["<1000g", "1000–1499g", "1500–2499g", "≥2500g", "Unknown"]
    ).fillna(0).astype(int)
    bw_total = bw_counts.sum()
    bw_tbl = pd.DataFrame({
        "Birth Weight Category": bw_counts.index,
        "n": bw_counts.values,
        "% of Total": [f"{(v / bw_total * 100):.1f}%" for v in bw_counts.values],
    })
    st.markdown(render_lancet_table(
        bw_tbl, title=f"{section_prefix}Distribution of neonates by birth weight category",
        footer_row=["Total", str(bw_total), "100.0%"],
        table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_BirthWeight"] = bw_tbl
    tc += 1
    
    # Visualization — birthweight bar
    bw_viz = bw_tbl[bw_tbl["Birth Weight Category"] != "Unknown"].copy()
    bw_colors = [LANCET_COLORS[1], LANCET_COLORS[9], LANCET_COLORS[3], LANCET_COLORS[2]]
    fig_bw = go.Figure(go.Bar(
        x=bw_viz["Birth Weight Category"],
        y=bw_viz["n"],
        marker_color=bw_colors[:len(bw_viz)],
        text=bw_viz["n"],
        textposition="outside",
    ))
    fig_bw = lancet_plotly_layout(
        fig_bw,
        title=f"Figure. {section_prefix}Birth weight distribution",
        xaxis_title="Birth Weight Category",
        yaxis_title="Number of Neonates",
        height=400,
    )
    st.plotly_chart(fig_bw, use_container_width=True)
    figures_for_export[f"{section_prefix}Birth_Weight"] = fig_bw

    # ── 9. Final Diagnosis ──
    st.markdown(f'<h4 class="report-section">{section_prefix}9. Final Diagnosis</h4>', unsafe_allow_html=True)
    
    diag_tbl, diag_footer = freq_table(df["Final Diagnosis"], "Final Diagnosis")
    st.markdown(render_lancet_table(
        diag_tbl, title=f"{section_prefix}Final diagnosis of admitted neonates",
        footer_row=diag_footer, table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Diagnosis"] = diag_tbl
    tc += 1
    
    # Top 10 diagnosis bar chart
    top10_diag = diag_tbl.head(10).copy()
    top10_diag["n"] = top10_diag["n"].astype(int)
    fig_diag = go.Figure(go.Bar(
        y=top10_diag["Final Diagnosis"][::-1],
        x=top10_diag["n"][::-1],
        orientation="h",
        marker_color=LANCET_COLORS[0],
        text=top10_diag["n"][::-1],
        textposition="outside",
    ))
    fig_diag = lancet_plotly_layout(
        fig_diag,
        title=f"Figure. {section_prefix}Top 10 final diagnoses",
        xaxis_title="Number of Neonates",
        height=420,
    )
    st.plotly_chart(fig_diag, use_container_width=True)
    figures_for_export[f"{section_prefix}Top10_Diagnoses"] = fig_diag

    # Top 10 diagnoses — mortality rate table
    top10_names = diag_tbl.head(10)["Final Diagnosis"].tolist()
    top10_df = df[df["Final Diagnosis"].isin(top10_names)]
    mort_by_diag = top10_df.groupby("Final Diagnosis").agg(
        Total=("Died", "size"),
        Deaths=("Died", "sum"),
    ).reset_index()
    mort_by_diag["Mortality (%)"] = (mort_by_diag["Deaths"] / mort_by_diag["Total"] * 100).round(1).astype(str) + "%"
    mort_by_diag = mort_by_diag.sort_values("Deaths", ascending=False)
    mort_by_diag.columns = ["Final Diagnosis", "Total (n)", "Deaths (n)", "Mortality (%)"]
    total_n = mort_by_diag["Total (n)"].sum()
    total_d = mort_by_diag["Deaths (n)"].sum()
    total_m = f"{(total_d / total_n * 100):.1f}%" if total_n > 0 else "0.0%"
    st.markdown(render_lancet_table(
        mort_by_diag,
        title=f"{section_prefix}Mortality rate by top 10 diagnoses",
        footer_row=["Total (Top 10)", str(total_n), str(total_d), total_m],
        table_num=tc,
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Diagnosis_Mortality"] = mort_by_diag
    tc += 1

    # ── 10. Hospital Outcome ──
    st.markdown(f'<h4 class="report-section">{section_prefix}10. Hospital Outcome (Combined 7-day & 28-day)</h4>', unsafe_allow_html=True)
    
    outcome_tbl, outcome_footer = freq_table(df["Combined_Outcome"], "Hospital Outcome")
    st.markdown(render_lancet_table(
        outcome_tbl, title=f"{section_prefix}Combined hospital outcome (7-day and 28-day status)",
        footer_row=outcome_footer, table_num=tc
    ), unsafe_allow_html=True)
    tables_for_export[f"T{tc}_Outcome"] = outcome_tbl
    tc += 1
    
    # Also show breakdowns
    col1, col2 = st.columns(2)
    with col1:
        s7_tbl, s7_footer = freq_table(df["Status_7d"], "Status at 7 Days")
        st.markdown(render_lancet_table(
            s7_tbl, title=f"{section_prefix}Status at 7 days",
            footer_row=s7_footer, table_num=tc
        ), unsafe_allow_html=True)
        tables_for_export[f"T{tc}_Status7d"] = s7_tbl
        tc += 1
    with col2:
        s28_tbl, s28_footer = freq_table(df["Status_28d"], "Status at 28 Days")
        st.markdown(render_lancet_table(
            s28_tbl, title=f"{section_prefix}Status at 28 days",
            footer_row=s28_footer, table_num=tc
        ), unsafe_allow_html=True)
        tables_for_export[f"T{tc}_Status28d"] = s28_tbl
        tc += 1
    
    # Outcome visualization
    outcome_colors = {
        "Discharged": LANCET_COLORS[2],
        "Died": LANCET_COLORS[1],
        "Self-Discharge / Runaway": LANCET_COLORS[9],
        "Referred Out": LANCET_COLORS[3],
        "Unknown": LANCET_COLORS[7],
    }
    fig_out = go.Figure(go.Pie(
        labels=outcome_tbl["Hospital Outcome"],
        values=outcome_tbl["n"],
        marker=dict(colors=[outcome_colors.get(x, LANCET_COLORS[7]) for x in outcome_tbl["Hospital Outcome"]]),
        textinfo="label+percent",
        hole=0.35,
    ))
    fig_out = lancet_plotly_layout(fig_out, title=f"Figure. {section_prefix}Hospital outcome", height=400)
    st.plotly_chart(fig_out, use_container_width=True)
    figures_for_export[f"{section_prefix}Hospital_Outcome"] = fig_out

    # Monthly mortality trend
    if df["Died"].any():
        monthly_deaths = df.groupby("Admission_Month").agg(
            Total=("Died", "size"),
            Deaths=("Died", "sum"),
        ).reset_index()
        monthly_deaths["Mortality_Rate"] = (monthly_deaths["Deaths"] / monthly_deaths["Total"] * 100).round(1)
        monthly_deaths["Month"] = monthly_deaths["Admission_Month"].astype(str)
        monthly_deaths = monthly_deaths.sort_values("Admission_Month")
        
        fig_mort = make_subplots(specs=[[{"secondary_y": True}]])
        fig_mort.add_trace(
            go.Bar(
                x=monthly_deaths["Month"], y=monthly_deaths["Deaths"],
                name="Deaths (n)", marker_color=LANCET_COLORS[1], opacity=0.7,
            ),
            secondary_y=False,
        )
        fig_mort.add_trace(
            go.Scatter(
                x=monthly_deaths["Month"], y=monthly_deaths["Mortality_Rate"],
                name="Mortality Rate (%)", mode="lines+markers",
                line=dict(color=LANCET_COLORS[0], width=2),
                marker=dict(size=8),
            ),
            secondary_y=True,
        )
        fig_mort = lancet_plotly_layout(
            fig_mort,
            title=f"Figure. {section_prefix}Monthly deaths and mortality rate",
            xaxis_title="Month",
            height=420,
        )
        fig_mort.update_yaxes(title_text="Number of Deaths", secondary_y=False)
        fig_mort.update_yaxes(title_text="Mortality Rate (%)", secondary_y=True)
        st.plotly_chart(fig_mort, use_container_width=True)
        figures_for_export[f"{section_prefix}Mortality_Trend"] = fig_mort

    return tables_for_export, tc, figures_for_export


def generate_excel_report(all_tables):
    """Export all tables to a formatted Excel workbook."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        
        # Formats
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#00468B",
            "font_color": "#FFFFFF",
            "border": 1,
            "font_name": "Arial",
            "font_size": 11,
        })
        cell_fmt = workbook.add_format({
            "border": 1,
            "font_name": "Arial",
            "font_size": 10,
        })
        
        for sheet_name, df_tbl in all_tables.items():
            # Truncate sheet name to 31 chars (Excel limit)
            safe_name = sheet_name[:31]
            df_tbl.to_excel(writer, sheet_name=safe_name, index=False, startrow=1)
            ws = writer.sheets[safe_name]
            
            # Write headers with formatting
            for col_idx, col_name in enumerate(df_tbl.columns):
                ws.write(0, col_idx, col_name, header_fmt)
            
            # Auto-fit column widths
            for col_idx, col_name in enumerate(df_tbl.columns):
                max_len = max(
                    df_tbl[col_name].astype(str).str.len().max(),
                    len(col_name)
                ) + 3
                ws.set_column(col_idx, col_idx, min(max_len, 50))
    
    output.seek(0)
    return output


def generate_figures_zip(all_figures):
    """Export all Plotly figures as PNG images inside a zip archive."""
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, fig in all_figures.items():
            # Clean filename: remove trailing spaces from section prefix
            clean_name = name.strip().replace(" ", "_").replace(".", "")
            png_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            zf.writestr(f"{clean_name}.png", png_bytes)
    zip_buffer.seek(0)
    return zip_buffer


# ──────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────────────────────

def main():
    # ── Header ──
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 0 0;">
        <h1 style="color: #00468B; font-family: Arial; margin-bottom: 0;">
            🏥 Neonatal Unit Monthly Report
        </h1>
        <p style="color: #666; font-size: 15px; margin-top: 4px;">
            Mbale Regional Referral Hospital &nbsp;|&nbsp; Automated Reporting Tool
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        st.markdown(
            "Upload the monthly neonatal logbook spreadsheet. "
            "Supports `.xlsx` (password-protected) and `.csv` formats."
        )
        
        file_type = st.radio("File format:", ["Excel (.xlsx)", "CSV (.csv)"], horizontal=True)
        
        password = None
        if file_type == "Excel (.xlsx)":
            password = st.text_input("Spreadsheet password:", type="password", help="Enter the password for the protected Excel file")
        
        uploaded_file = st.file_uploader(
            "Choose file", type=["xlsx", "csv"],
            help="Upload the neonatal logbook data file"
        )
        
        st.divider()
        st.markdown("### ⚙️ Report Settings")
        
        show_deaths_only = st.checkbox("Section 11: Deaths only report", value=True)
        show_lbw = st.checkbox("Section 12: Low birth weight sub-reports", value=True)
    
    # ── Load & process data ──
    df = None
    
    if uploaded_file is not None:
        if file_type == "CSV (.csv)":
            try:
                df = pd.read_csv(uploaded_file, encoding="latin-1")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return
        else:
            # Handle password-protected Excel
            if password:
                try:
                    import msoffcrypto
                    decrypted = io.BytesIO()
                    ms_file = msoffcrypto.OfficeFile(uploaded_file)
                    ms_file.load_key(password=password)
                    ms_file.decrypt(decrypted)
                    decrypted.seek(0)
                    df = pd.read_excel(decrypted, sheet_name="Logbook", engine="openpyxl")
                except Exception as e:
                    err_name = type(e).__name__
                    err_msg = str(e).lower()
                    if "invalidkey" in err_name.lower() or "could not be decrypted" in err_msg:
                        st.error("Incorrect password. Please check the password and try again.")
                    else:
                        st.error(f"Error reading file: {str(e)}")
                    return
            else:
                try:
                    df = pd.read_excel(uploaded_file, sheet_name="Logbook", engine="openpyxl")
                except Exception as e:
                    err_name = type(e).__name__
                    err_msg = str(e).lower()
                    if "zip" in err_msg or "encrypt" in err_msg or "badzipfile" in err_name.lower():
                        st.error("This file appears to be password-protected. Please enter the spreadsheet password in the sidebar.")
                    else:
                        st.error(f"Error reading file: {str(e)}")
                    return
    else:
        # Demo mode: check for local CSV
        demo_path = os.path.join(os.path.dirname(__file__), "data", "Logbook.csv")
        if os.path.exists(demo_path):
            df = pd.read_csv(demo_path, encoding="latin-1")
            st.info("📊 Running in demo mode with sample data. Upload your own file via the sidebar.")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px; color: #666;">
                <h3 style="color: #00468B;">Welcome</h3>
                <p>Please upload the neonatal logbook spreadsheet using the sidebar to generate the monthly report.</p>
                <p style="font-size: 13px;">
                    Supported formats: <strong>.xlsx</strong> (password-protected) or <strong>.csv</strong> export of the Logbook tab.
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
    
    # ── Clean data ──
    df = clean_data(df)
    valid_df = df.dropna(subset=["Admission_Month"])
    
    # ── Month filter ──
    available_months = sorted(valid_df["Admission_Month"].dropna().unique())
    month_labels = [str(m) for m in available_months]
    
    with st.sidebar:
        st.divider()
        st.markdown("### 📅 Month Filter")
        filter_mode = st.radio(
            "Report scope:", 
            ["All months (cumulative)", "Select specific month(s)"],
            horizontal=False,
        )
        
        if filter_mode == "Select specific month(s)":
            selected_months = st.multiselect(
                "Choose month(s):", month_labels, default=month_labels
            )
            filtered = valid_df[valid_df["Admission_Month"].astype(str).isin(selected_months)]
        else:
            filtered = valid_df.copy()
            selected_months = month_labels
    
    # ── Summary metrics ──
    n_records = len(filtered)
    n_deaths = filtered["Died"].sum()
    mortality_pct = (n_deaths / n_records * 100) if n_records > 0 else 0
    n_lbw = (filtered["BW_Category"].isin(["<1000g", "1000–1499g"])).sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_records:,}</div>
            <div class="metric-label">Total Admissions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{int(n_deaths):,}</div>
            <div class="metric-label">Deaths</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{mortality_pct:.1f}%</div>
            <div class="metric-label">Mortality Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{int(n_lbw):,}</div>
            <div class="metric-label">Very/Extremely LBW</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ── SECTION A: All admissions ──
    all_tables = {}
    all_figures = {}

    st.markdown('<h2 style="color: #00468B; text-align: center;">Section A — All Admissions</h2>', unsafe_allow_html=True)
    tables_a, tc, figs_a = generate_report_section(filtered, section_prefix="A. ", table_counter_start=1)
    all_tables.update(tables_a)
    all_figures.update(figs_a)

    # ── SECTION B: Deaths only (Item 11) ──
    if show_deaths_only:
        st.divider()
        st.markdown('<h2 style="color: #AD002A; text-align: center;">Section B — Deaths Only</h2>', unsafe_allow_html=True)
        deaths_df = filtered[filtered["Died"]].copy()
        st.markdown(f"*Filtered to {len(deaths_df)} neonatal deaths out of {n_records} total admissions.*")
        tables_b, tc, figs_b = generate_report_section(deaths_df, section_prefix="B. ", table_counter_start=tc)
        all_tables.update(tables_b)
        all_figures.update(figs_b)
    
    # ── SECTION C: Low birth weight sub-reports (Item 12) ──
    if show_lbw:
        st.divider()
        st.markdown('<h2 style="color: #925E9F; text-align: center;">Section C — Extremely Low Birth Weight (&lt;1000g)</h2>', unsafe_allow_html=True)
        elbw = filtered[filtered["BW_Category"] == "<1000g"].copy()
        st.markdown(f"*Filtered to {len(elbw)} neonates with birth weight <1000g.*")
        tables_c1, tc, figs_c1 = generate_report_section(elbw, section_prefix="C1. ", table_counter_start=tc)
        all_tables.update(tables_c1)
        all_figures.update(figs_c1)

        st.divider()
        st.markdown('<h2 style="color: #925E9F; text-align: center;">Section C — Very Low Birth Weight (1000–1499g)</h2>', unsafe_allow_html=True)
        vlbw = filtered[filtered["BW_Category"] == "1000–1499g"].copy()
        st.markdown(f"*Filtered to {len(vlbw)} neonates with birth weight 1000–1499g.*")
        tables_c2, tc, figs_c2 = generate_report_section(vlbw, section_prefix="C2. ", table_counter_start=tc)
        all_tables.update(tables_c2)
        all_figures.update(figs_c2)
    
    # ── Download button ──
    st.divider()
    st.markdown('<h3 style="color: #00468B;">📥 Download Full Report</h3>', unsafe_allow_html=True)
    
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        excel_data = generate_excel_report(all_tables)
        st.download_button(
            label="⬇️  Download Excel Report",
            data=excel_data,
            file_name=f"Neonatal_Report_{'_'.join(selected_months) if len(selected_months) <= 3 else 'cumulative'}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dl_col2:
        if all_figures:
            zip_data = generate_figures_zip(all_figures)
            st.download_button(
                label="⬇️  Download All Graphs (ZIP)",
                data=zip_data,
                file_name=f"Neonatal_Graphs_{'_'.join(selected_months) if len(selected_months) <= 3 else 'cumulative'}.zip",
                mime="application/zip",
            )

    # ── Footer ──
    st.divider()
    st.markdown("""
    <div style="text-align: center; font-size: 12px; color: #999; padding: 10px;">
        Neonatal Reporting Tool v1.0 &nbsp;|&nbsp; Mbale Regional Referral Hospital<br>
        Developed by Raymond R. Wayesu &nbsp;|&nbsp; Elgon Centre for Health Research & Innovation (ELCHRI)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
