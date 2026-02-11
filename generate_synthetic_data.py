"""
Gaussian Copula-based synthetic neonatal data generator.

Uses the copulas library (GaussianMultivariate) to learn the full joint
distribution from the real Mbale RRH neonatal logbook data, preserving:
  - Marginal distributions of every variable
  - Inter-variable correlations (diagnosis↔birth weight, diagnosis↔outcome, etc.)
  - Realistic missing-data patterns

Methodology:
  Rubin (1993) framework for synthetic data generation.  The Gaussian copula
  captures the dependence structure while allowing each marginal to retain
  its empirical distribution.  Categorical variables are integer-encoded,
  modelled jointly, then decoded back via nearest-valid-code rounding.

References:
  - Rubin, D.B. (1993). Statistical Disclosure Limitation. J. Official Statistics.
  - Patki et al. (2016). The Synthetic Data Vault. IEEE DSAA.
  - copulas library: https://github.com/sdv-dev/Copulas

Usage:
  1. Place the real (unencrypted) Logbook.csv in data/real_Logbook.csv
  2. Run: python generate_synthetic_data.py
  3. Output: data/Logbook.csv (synthetic, safe to commit)

The real data file is never committed (excluded by .gitignore).
"""

import pandas as pd
import numpy as np
from copulas.multivariate import GaussianMultivariate
from datetime import datetime, timedelta
import warnings
import sys
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ──
N_SYNTHETIC = 2000
REAL_DATA_PATH = "data/real_Logbook.csv"
OUTPUT_PATH = "data/Logbook.csv"


# ══════════════════════════════════════════════════════════════
# STEP 1: Load and clean real data
# ══════════════════════════════════════════════════════════════

def load_and_clean(path):
    """Load real data and standardise inconsistent values."""
    df = pd.read_csv(path, encoding="latin-1")
    print(f"Loaded {len(df)} real records from {path}")

    # ── Fix case inconsistencies in categorical columns ──
    # Mode of delivery
    mode_map = {
        "Spontaneous Vaginal Delivery": "Spontaneous Vaginal Delivery",
        "spontaneous Vaginal Delivery": "Spontaneous Vaginal Delivery",
        "Caesarean Section": "Caesarean Section",
        "caesarean Section": "Caesarean Section",
    }
    df["Mode of Delivery"] = df["Mode of Delivery"].map(
        lambda x: mode_map.get(x, x) if pd.notna(x) else x
    )

    # HIV status
    hiv_map = {"TR": "TR", "TRR": "TRR", "Unknown": "Unknown", "unknown": "Unknown"}
    df["HIV Status Code"] = df["HIV Status Code"].map(
        lambda x: hiv_map.get(x, x) if pd.notna(x) else x
    )

    # How many babies
    babies_map = {
        "Singleton": "Singleton", "singleton": "Singleton",
        "Twin": "Twin", "Triplet": "Triplet",
    }
    df["How many babies born"] = df["How many babies born"].map(
        lambda x: babies_map.get(x, x) if pd.notna(x) else x
    )

    # Final diagnosis
    diag_map = {
        "bacterial Sepsis of newborn, unspecified": "Bacterial Sepsis of newborn, unspecified",
        "neonatal Jaundice, unspecified": "Neonatal Jaundice, unspecified",
        "other": "Other",
    }
    df["Final Diagnosis"] = df["Final Diagnosis"].replace(diag_map)

    # Sex — drop invalid entries
    df.loc[~df["Sex"].isin(["Male", "Female"]) & df["Sex"].notna(), "Sex"] = np.nan

    # Birth weight → numeric
    df["Birth Weight (Kg)"] = pd.to_numeric(df["Birth Weight (Kg)"], errors="coerce")

    return df


# ══════════════════════════════════════════════════════════════
# STEP 2: Fit Gaussian Copula on core clinical variables
# ══════════════════════════════════════════════════════════════

# Variables modelled jointly by the copula
COPULA_VARS = {
    "Birth Weight (Kg)": "continuous",
    "Mother's Age in years": "continuous",
    "Final Diagnosis": "categorical",
    "HIV Status Code": "categorical",
    "Sex": "categorical",
    "Mode of Delivery": "categorical",
    "How many babies born": "categorical",
    "Status at 7 Days - Outcome": "categorical",
    "Status at 28 Days - Outcome": "categorical",
}


def encode_for_copula(df):
    """Integer-encode categoricals and return (numeric_df, encoding_maps)."""
    # Work with complete-ish rows only (has an admission date)
    complete = df[df["Date of Admission"].notna()].copy()

    encodings = {}
    for col, ctype in COPULA_VARS.items():
        if ctype == "categorical":
            # Treat NaN as an explicit category (preserves missingness pattern)
            complete[col] = complete[col].fillna("_MISSING_")
            cats = complete[col].value_counts().index.tolist()
            to_code = {cat: i for i, cat in enumerate(cats)}
            encodings[col] = {
                "to_code": to_code,
                "to_label": {v: k for k, v in to_code.items()},
            }
            complete[col] = complete[col].map(to_code).astype(float)

    # Select only copula columns, drop remaining NaN (continuous cols)
    copula_cols = list(COPULA_VARS.keys())
    numeric = complete[copula_cols].dropna()
    print(f"Copula fitted on {len(numeric)} complete rows × {len(copula_cols)} variables")

    return numeric, encodings


def fit_copula(numeric_df):
    """Fit GaussianMultivariate copula."""
    model = GaussianMultivariate()
    model.fit(numeric_df)
    return model


def sample_and_decode(model, encodings, n):
    """Sample from the fitted copula and decode categoricals."""
    raw = model.sample(n)

    for col, ctype in COPULA_VARS.items():
        if ctype == "continuous":
            # Clip to realistic ranges
            if col == "Birth Weight (Kg)":
                raw[col] = raw[col].clip(0.45, 6.0).round(3)
            elif col == "Mother's Age in years":
                raw[col] = raw[col].clip(14, 50).round(0).astype(int)
        else:
            n_cats = len(encodings[col]["to_code"])
            raw[col] = (
                raw[col].round().clip(0, n_cats - 1).astype(int)
                .map(encodings[col]["to_label"])
            )
            # Restore NaN for "_MISSING_"
            raw[col] = raw[col].replace("_MISSING_", np.nan)

    return raw


# ══════════════════════════════════════════════════════════════
# STEP 3: Learn auxiliary distributions from real data
# ══════════════════════════════════════════════════════════════

def learn_auxiliary(df):
    """Learn conditional and marginal distributions for non-copula variables."""
    complete = df[df["Date of Admission"].notna()].copy()
    aux = {}

    # ── Monthly admission volumes ──
    complete["_adm_date"] = pd.to_datetime(complete["Date of Admission"], format="%d.%m.%y", errors="coerce")
    complete["_month"] = complete["_adm_date"].dt.to_period("M")
    month_counts = complete["_month"].value_counts().sort_index()
    month_props = (month_counts / month_counts.sum()).to_dict()
    aux["month_distribution"] = month_props

    # Date ranges per month
    month_ranges = {}
    for m in month_counts.index:
        dates_in_month = complete.loc[complete["_month"] == m, "_adm_date"]
        month_ranges[m] = (dates_in_month.min(), dates_in_month.max())
    aux["month_ranges"] = month_ranges

    # ── Place of Birth distribution ──
    pob = complete["Place of Birth"].dropna().value_counts(normalize=True)
    aux["place_of_birth"] = pob.to_dict()

    # ── Referral From conditional on Place of Birth ──
    # In the real data, Place of Birth and Referral From are usually the same
    both = complete[["Place of Birth", "Referral From"]].dropna()
    same_pct = (both["Place of Birth"] == both["Referral From"]).mean()
    aux["referral_same_as_pob"] = same_pct
    # When different, learn the referral distribution
    diff = both[both["Place of Birth"] != both["Referral From"]]
    if len(diff) > 0:
        aux["referral_when_different"] = diff["Referral From"].value_counts(normalize=True).to_dict()
    else:
        aux["referral_when_different"] = aux["place_of_birth"]

    # ── District distribution ──
    dist = complete["Address (District)"].dropna().value_counts(normalize=True)
    aux["districts"] = dist.to_dict()

    # ── Village distribution (per district, top villages) ──
    village_by_district = {}
    for d in dist.index:
        vils = complete.loc[complete["Address (District)"] == d, "Address (Village)"].dropna()
        if len(vils) > 0:
            village_by_district[d] = vils.value_counts(normalize=True).to_dict()
    aux["villages_by_district"] = village_by_district

    # ── Reason for Admission conditional on Final Diagnosis ──
    reason_by_diag = {}
    for diag in complete["Final Diagnosis"].dropna().unique():
        reasons = complete.loc[
            complete["Final Diagnosis"] == diag, "Reason for Admission"
        ].dropna()
        if len(reasons) > 0:
            reason_by_diag[diag] = reasons.value_counts(normalize=True).to_dict()
    aux["reasons_by_diagnosis"] = reason_by_diag

    # Overall reason distribution (fallback)
    aux["reasons_overall"] = (
        complete["Reason for Admission"].dropna().value_counts(normalize=True).to_dict()
    )

    # ── Missing data rate per column ──
    n_complete = len(complete)
    aux["missing_rates"] = {
        "Address (District)": complete["Address (District)"].isna().mean(),
        "Address (Village)": complete["Address (Village)"].isna().mean(),
        "Reason for Admission": complete["Reason for Admission"].isna().mean(),
    }

    # ── Skeleton row proportion (rows with only IP No.) ──
    skeleton = df[df["Date of Admission"].isna()]
    aux["skeleton_proportion"] = len(skeleton) / len(df) if len(df) > 0 else 0.15

    # ── Birth delay: days between birth and admission ──
    complete["_birth_date"] = pd.to_datetime(complete["Date of Birth"], format="%d.%m.%y", errors="coerce")
    delays = (complete["_adm_date"] - complete["_birth_date"]).dt.days.dropna()
    delays = delays[(delays >= 0) & (delays < 60)]  # realistic range
    aux["birth_delay_values"] = delays.values

    # ── Outcome date delays ──
    for status_col, date_col in [
        ("Status at 7 Days - Outcome", "Status at 7 Days - Date"),
        ("Status at 28 Days - Outcome", "Status at 28 Days - Date"),
    ]:
        has_outcome = complete[complete[status_col].notna()]
        outcome_dates = pd.to_datetime(has_outcome[date_col], format="%d.%m.%y", errors="coerce")
        outcome_delays = (outcome_dates - has_outcome["_adm_date"]).dt.days.dropna()
        outcome_delays = outcome_delays[(outcome_delays >= 0) & (outcome_delays < 60)]
        aux[f"{status_col}_delays"] = outcome_delays.values if len(outcome_delays) > 0 else np.array([3])

    return aux


# ══════════════════════════════════════════════════════════════
# STEP 4: Assemble full synthetic dataset
# ══════════════════════════════════════════════════════════════

def _sample_from_dict(dist_dict, n=1):
    """Sample n values from a {value: probability} dictionary."""
    keys = list(dist_dict.keys())
    probs = np.array(list(dist_dict.values()), dtype=float)
    probs /= probs.sum()
    return np.random.choice(keys, size=n, p=probs)


def _random_time():
    """Random time string HH:MM."""
    return f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}"


def assemble_synthetic(copula_samples, aux, n_total):
    """Combine copula samples with auxiliary variables into full records."""

    n_complete = len(copula_samples)
    n_skeleton = n_total - n_complete

    rows = []
    ip_start = 36474

    # ── Assign months to each copula row ──
    month_list = list(aux["month_distribution"].keys())
    month_probs = np.array([aux["month_distribution"][m] for m in month_list])
    month_probs /= month_probs.sum()
    assigned_months = np.random.choice(month_list, size=n_complete, p=month_probs)

    for i in range(n_complete):
        cs = copula_samples.iloc[i]
        month = assigned_months[i]
        m_start, m_end = aux["month_ranges"][month]

        # Admission date — random within the month
        delta = max((m_end - m_start).days, 1)
        adm_date = m_start + timedelta(days=int(np.random.randint(0, delta + 1)))

        # Birth date — realistic delay before admission
        if len(aux["birth_delay_values"]) > 0:
            delay = int(np.random.choice(aux["birth_delay_values"]))
        else:
            delay = 0
        birth_date = adm_date - timedelta(days=delay)

        # Place of birth
        pob = _sample_from_dict(aux["place_of_birth"])[0]

        # Referral from (usually same as place of birth)
        if np.random.random() < aux["referral_same_as_pob"]:
            referral = pob
        else:
            referral = _sample_from_dict(aux["referral_when_different"])[0]

        # District
        if np.random.random() < aux["missing_rates"]["Address (District)"]:
            district = ""
            village = ""
        else:
            district = _sample_from_dict(aux["districts"])[0]
            # Village
            if np.random.random() < aux["missing_rates"]["Address (Village)"]:
                village = ""
            elif district in aux["villages_by_district"]:
                village = _sample_from_dict(aux["villages_by_district"][district])[0]
            else:
                village = ""

        # Reason for admission
        diagnosis = cs["Final Diagnosis"]
        if np.random.random() < aux["missing_rates"]["Reason for Admission"]:
            reason = ""
        elif pd.notna(diagnosis) and diagnosis in aux["reasons_by_diagnosis"]:
            reason = _sample_from_dict(aux["reasons_by_diagnosis"][diagnosis])[0]
        elif aux["reasons_overall"]:
            reason = _sample_from_dict(aux["reasons_overall"])[0]
        else:
            reason = ""

        # Outcome dates
        s7_date = ""
        if pd.notna(cs["Status at 7 Days - Outcome"]):
            delays = aux["Status at 7 Days - Outcome_delays"]
            d = int(np.random.choice(delays)) if len(delays) > 0 else 3
            s7_date = (adm_date + timedelta(days=d)).strftime("%d.%m.%y")

        s28_date = ""
        if pd.notna(cs["Status at 28 Days - Outcome"]):
            delays = aux["Status at 28 Days - Outcome_delays"]
            d = int(np.random.choice(delays)) if len(delays) > 0 else 10
            s28_date = (adm_date + timedelta(days=d)).strftime("%d.%m.%y")

        rows.append({
            "IP No.": ip_start + i,
            "Address (District)": district,
            "Address (Village)": village,
            "Mother's Age in years": cs["Mother's Age in years"] if pd.notna(cs["Mother's Age in years"]) else "",
            "HIV Status Code": cs["HIV Status Code"] if pd.notna(cs["HIV Status Code"]) else "",
            "Sex": cs["Sex"] if pd.notna(cs["Sex"]) else "",
            "Date of Birth": birth_date.strftime("%d.%m.%y"),
            "Time of Birth": _random_time(),
            "Place of Birth": pob,
            "Mode of Delivery": cs["Mode of Delivery"] if pd.notna(cs["Mode of Delivery"]) else "",
            "Birth Weight (Kg)": cs["Birth Weight (Kg)"] if pd.notna(cs["Birth Weight (Kg)"]) else "",
            "How many babies born": cs["How many babies born"] if pd.notna(cs["How many babies born"]) else "",
            "Source of warmth": "",
            "BCG ": "",
            "Polio": "",
            "Date of Admission": adm_date.strftime("%d.%m.%y"),
            "Time of Admission": _random_time(),
            "Reason for Admission": reason,
            "Blood Transfusion": "",
            "Final Diagnosis": cs["Final Diagnosis"] if pd.notna(cs["Final Diagnosis"]) else "",
            "Other Diagnosis": "",
            "Referral From": referral,
            "Discharge Weight (kg)": "",
            "Status at 7 Days - Outcome": cs["Status at 7 Days - Outcome"] if pd.notna(cs["Status at 7 Days - Outcome"]) else "",
            "Status at 7 Days - Date": s7_date,
            "Status at 28 Days - Outcome": cs["Status at 28 Days - Outcome"] if pd.notna(cs["Status at 28 Days - Outcome"]) else "",
            "Status at 28 Days - Date": s28_date,
            "Service Provider": "",
        })

    # ── Skeleton rows (empty except IP No., matching real data proportion) ──
    for i in range(n_skeleton):
        rows.append({
            "IP No.": ip_start + n_complete + i,
            "Address (District)": "",
            "Address (Village)": "",
            "Mother's Age in years": "",
            "HIV Status Code": "",
            "Sex": "",
            "Date of Birth": "",
            "Time of Birth": "",
            "Place of Birth": "",
            "Mode of Delivery": "",
            "Birth Weight (Kg)": "",
            "How many babies born": "",
            "Source of warmth": "",
            "BCG ": "",
            "Polio": "",
            "Date of Admission": "",
            "Time of Admission": "",
            "Reason for Admission": "",
            "Blood Transfusion": "",
            "Final Diagnosis": "",
            "Other Diagnosis": "",
            "Referral From": "",
            "Discharge Weight (kg)": "",
            "Status at 7 Days - Outcome": "",
            "Status at 7 Days - Date": "",
            "Status at 28 Days - Outcome": "",
            "Status at 28 Days - Date": "",
            "Service Provider": "",
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# STEP 5: Validate synthetic vs real
# ══════════════════════════════════════════════════════════════

def validate(real_df, synth_df):
    """Print comparative statistics for validation."""
    real = real_df[real_df["Date of Admission"].notna()].copy()
    synth = synth_df[synth_df["Date of Admission"] != ""].copy()

    print("\n" + "=" * 60)
    print("VALIDATION: Real vs Synthetic")
    print("=" * 60)

    print(f"\nRecord counts — Real: {len(real)}, Synthetic: {len(synth)}")

    # Birth weight
    real_bw = pd.to_numeric(real["Birth Weight (Kg)"], errors="coerce").dropna()
    synth_bw = pd.to_numeric(synth["Birth Weight (Kg)"], errors="coerce").dropna()
    print(f"\nBirth Weight (Kg):")
    print(f"  Real  — mean={real_bw.mean():.3f}, std={real_bw.std():.3f}, median={real_bw.median():.3f}")
    print(f"  Synth — mean={synth_bw.mean():.3f}, std={synth_bw.std():.3f}, median={synth_bw.median():.3f}")

    # Mother's age
    real_age = pd.to_numeric(real["Mother's Age in years"], errors="coerce").dropna()
    synth_age = pd.to_numeric(synth["Mother's Age in years"], errors="coerce").dropna()
    print(f"\nMother's Age:")
    print(f"  Real  — mean={real_age.mean():.1f}, std={real_age.std():.1f}")
    print(f"  Synth — mean={synth_age.mean():.1f}, std={synth_age.std():.1f}")

    # Key categorical comparisons
    for col in ["Final Diagnosis", "HIV Status Code", "Sex", "Mode of Delivery",
                "How many babies born", "Status at 7 Days - Outcome", "Status at 28 Days - Outcome"]:
        print(f"\n{col}:")
        real_vc = real[col].fillna("_Missing_").value_counts(normalize=True).head(8)
        synth_vc = synth[col].replace("", np.nan).fillna("_Missing_").value_counts(normalize=True).head(8)
        combined = pd.DataFrame({"Real %": (real_vc * 100).round(1), "Synth %": (synth_vc * 100).round(1)})
        print(combined.to_string())

    # Mortality
    real["_died"] = real["Status at 7 Days - Outcome"].isin(["DD"]) | real["Status at 28 Days - Outcome"].isin(["DD"])
    synth["_died"] = synth["Status at 7 Days - Outcome"].isin(["DD"]) | synth["Status at 28 Days - Outcome"].isin(["DD"])
    real_mort = real["_died"].mean() * 100
    synth_mort = synth["_died"].mean() * 100
    print(f"\nOverall mortality — Real: {real_mort:.1f}%, Synthetic: {synth_mort:.1f}%")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # Check for real data
    if not os.path.exists(REAL_DATA_PATH):
        print(f"ERROR: Real data not found at {REAL_DATA_PATH}")
        print(f"Please copy the real Logbook.csv to {REAL_DATA_PATH} first.")
        sys.exit(1)

    # Step 1: Load and clean
    real_df = load_and_clean(REAL_DATA_PATH)

    # Step 2: Encode and fit copula
    numeric_df, encodings = encode_for_copula(real_df)
    model = fit_copula(numeric_df)
    print("Gaussian copula fitted successfully.")

    # Step 3: Learn auxiliary distributions
    aux = learn_auxiliary(real_df)

    # Step 4: Sample from copula and assemble full dataset
    n_skeleton = int(round(N_SYNTHETIC * aux["skeleton_proportion"]))
    n_complete = N_SYNTHETIC - n_skeleton
    print(f"\nGenerating {n_complete} complete + {n_skeleton} skeleton = {N_SYNTHETIC} total rows...")

    copula_samples = sample_and_decode(model, encodings, n_complete)
    synth_df = assemble_synthetic(copula_samples, aux, N_SYNTHETIC)

    # Step 5: Save
    synth_df.to_csv(OUTPUT_PATH, index=False, encoding="latin-1")
    print(f"\nSynthetic data saved to {OUTPUT_PATH}")

    # Step 6: Validate
    validate(real_df, synth_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
