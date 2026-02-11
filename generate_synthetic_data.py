"""
Generate synthetic neonatal logbook data preserving the statistical
properties of the real Mbale RRH dataset without any real patient information.

This script produces a demo Logbook.csv with ~2000 rows that mirrors:
- Monthly admission volumes
- Birth weight distributions (bimodal: preterm + normal)
- Diagnosis frequencies and their correlation with birth weight
- Outcome rates by diagnosis
- Facility level distributions
- Demographic distributions (sex, HIV status, delivery mode, etc.)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

N_RECORDS = 2000
N_SKELETON = 310  # ~15.5% empty rows
N_COMPLETE = N_RECORDS - N_SKELETON

# ── Synthetic facility names ──
FACILITIES_RRH = ["Mbale RRH"]
FACILITIES_GH = ["Tororo G.H", "Pallisa G.H", "Iganga G.H"]
FACILITIES_DH = ["Budaka D.H", "Sironko D.H", "Manafwa D.H"]
FACILITIES_HCIV = [
    "Namatala HCIV", "Busiu HCIV", "Bufumbo HCIV", "Budaka HCIV",
    "Budadiri HCIV", "Lwala HCIV", "Busolwe HCIV", "Kumi HCIV",
]
FACILITIES_HCIII = [
    "Namanyonyi HCIII", "Bukonde HCIII", "Bukhalu HCIII", "Nabiganda HCIII",
    "Buyobo HCIII", "Magale HCIII", "Nampanga HCIII", "Buwagogo HCIII",
    "Bukigai HCIII", "Nakaloke HCIII", "Busano HCIII", "Lwasso HCIII",
]
FACILITIES_HCII = ["Busamaga HCII", "Namakwekwe HCII", "Bunghokho HCII"]
FACILITIES_OTHER = ["BBA", "Home"]
ALL_FACILITIES = (
    FACILITIES_RRH * 40 + FACILITIES_GH * 4 + FACILITIES_DH * 3 +
    FACILITIES_HCIV * 8 + FACILITIES_HCIII * 4 + FACILITIES_HCII * 1 +
    FACILITIES_OTHER * 4
)

# ── Districts ──
DISTRICTS = {
    "Mbale": 0.447, "Sironko": 0.064, "Bukedea": 0.053, "Budaka": 0.045,
    "Manafwa": 0.028, "Tororo": 0.026, "Bulambuli": 0.023,
    "Namisindwa": 0.021, "Butebo": 0.020, "Kibuku": 0.020,
    "Pallisa": 0.020, "Bududa": 0.015, "Butaleja": 0.015,
    "Busia": 0.009, "Kapchorwa": 0.007,
}
# Normalize and add "Other"
total_p = sum(DISTRICTS.values())
DISTRICTS["Other District"] = 1.0 - total_p
district_names = list(DISTRICTS.keys())
district_probs = list(DISTRICTS.values())

# ── Synthetic village names ──
VILLAGES = [
    "Nakaloke", "Namakwekwe", "Busamaga", "Malukhu", "Nabumali",
    "Nkoma", "Wanale", "Bugema", "Industrial", "Bukhaweka",
    "Bumasifwa", "Busoba", "Bunghokho", "Lukhonge", "Namabasa",
    "Namagumba", "Bukasakya", "Bukonde", "Lwangoli", "Simu",
]

# ── Diagnoses with birth-weight and mortality parameters ──
DIAGNOSES = {
    "Bacterial Sepsis of newborn, unspecified": {
        "freq": 0.250, "bw_mean": 2.84, "bw_std": 0.66, "mortality_7d": 0.115, "mortality_28d": 0.08,
    },
    "Preterm - other (28 - <37 weeks)": {
        "freq": 0.225, "bw_mean": 1.87, "bw_std": 0.34, "mortality_7d": 0.223, "mortality_28d": 0.12,
    },
    "Other": {
        "freq": 0.185, "bw_mean": 2.70, "bw_std": 0.80, "mortality_7d": 0.100, "mortality_28d": 0.06,
    },
    "Preterm - extreme ( Less than 28 weeks by Ballard)": {
        "freq": 0.150, "bw_mean": 1.27, "bw_std": 0.74, "mortality_7d": 0.639, "mortality_28d": 0.30,
    },
    "Hypoxic ischemic encephalopathy [HIE], unspecified": {
        "freq": 0.069, "bw_mean": 3.10, "bw_std": 0.56, "mortality_7d": 0.435, "mortality_28d": 0.15,
    },
    "Neonatal Jaundice, unspecified": {
        "freq": 0.027, "bw_mean": 2.87, "bw_std": 0.61, "mortality_7d": 0.227, "mortality_28d": 0.10,
    },
    "Meconium aspiration syndrome": {
        "freq": 0.021, "bw_mean": 3.16, "bw_std": 0.46, "mortality_7d": 0.125, "mortality_28d": 0.08,
    },
    "Well baby": {
        "freq": 0.020, "bw_mean": 3.06, "bw_std": 0.80, "mortality_7d": 0.0, "mortality_28d": 0.0,
    },
    "Respiratory distress syndrome (RDS)": {
        "freq": 0.020, "bw_mean": 2.66, "bw_std": 0.81, "mortality_7d": 0.385, "mortality_28d": 0.15,
    },
    "Meningitis unspecified": {
        "freq": 0.010, "bw_mean": 3.06, "bw_std": 0.46, "mortality_7d": 0.500, "mortality_28d": 0.15,
    },
    "Gastroschisis": {
        "freq": 0.009, "bw_mean": 2.11, "bw_std": 0.49, "mortality_7d": 0.400, "mortality_28d": 0.20,
    },
    "Low birth weight - Other (1000g - 2499g)": {
        "freq": 0.007, "bw_mean": 1.93, "bw_std": 0.46, "mortality_7d": 0.150, "mortality_28d": 0.10,
    },
    "Hydrocephalus, unspecified": {
        "freq": 0.004, "bw_mean": 2.80, "bw_std": 0.50, "mortality_7d": 0.200, "mortality_28d": 0.10,
    },
    "Spina bifida, unspecified": {
        "freq": 0.003, "bw_mean": 2.70, "bw_std": 0.50, "mortality_7d": 0.200, "mortality_28d": 0.10,
    },
}

REASONS = {
    "Bacterial Sepsis of newborn, unspecified": ["Fever", "Sepsis", "DIB", "Hypothermia"],
    "Preterm - other (28 - <37 weeks)": ["Prematurity", "LBW", "Prematurity + LBW"],
    "Other": ["DIB", "BA", "Convulsions", "Fever", "Other"],
    "Preterm - extreme ( Less than 28 weeks by Ballard)": ["Prematurity", "Extreme Prematurity", "LBW"],
    "Hypoxic ischemic encephalopathy [HIE], unspecified": ["BA", "Birth Asphyxia", "BA + Convulsions"],
    "Neonatal Jaundice, unspecified": ["Jaundice", "Neonatal Jaundice"],
    "Meconium aspiration syndrome": ["MAS", "DIB + MAS"],
    "Well baby": ["Observation", "Mother unwell", "Well baby"],
    "Respiratory distress syndrome (RDS)": ["DIB", "RDS", "Respiratory Distress"],
    "Meningitis unspecified": ["Fever", "Convulsions", "Fever + Convulsions"],
    "Gastroschisis": ["Gastroschisis", "Abdominal wall defect"],
    "Low birth weight - Other (1000g - 2499g)": ["LBW", "Low birth weight"],
    "Hydrocephalus, unspecified": ["Big head", "Hydrocephalus"],
    "Spina bifida, unspecified": ["Spina bifida", "Neural tube defect"],
}

# ── Monthly admission distribution (Jul 2025 - Jan 2026) ──
MONTHS = [
    (datetime(2025, 7, 1), datetime(2025, 7, 31), 0.051),
    (datetime(2025, 8, 1), datetime(2025, 8, 31), 0.170),
    (datetime(2025, 9, 1), datetime(2025, 9, 30), 0.158),
    (datetime(2025, 10, 1), datetime(2025, 10, 31), 0.172),
    (datetime(2025, 11, 1), datetime(2025, 11, 30), 0.155),
    (datetime(2025, 12, 1), datetime(2025, 12, 31), 0.169),
    (datetime(2026, 1, 1), datetime(2026, 1, 22), 0.125),
]


def random_date_in_range(start, end, n=1):
    """Generate n random dates between start and end."""
    delta = (end - start).days
    return [start + timedelta(days=int(np.random.randint(0, delta + 1))) for _ in range(n)]


def random_time():
    """Generate a random time (roughly uniform across 24h)."""
    h = np.random.randint(0, 24)
    m = np.random.randint(0, 60)
    return f"{h:02d}:{m:02d}"


def generate_synthetic_data():
    rows = []
    ip_start = 36474

    # Assign complete rows to months
    month_assignments = []
    for start, end, prop in MONTHS:
        n_month = int(round(prop * N_COMPLETE))
        month_assignments.extend([(start, end)] * n_month)
    # Adjust to exactly N_COMPLETE
    while len(month_assignments) < N_COMPLETE:
        month_assignments.append((MONTHS[-1][0], MONTHS[-1][1]))
    month_assignments = month_assignments[:N_COMPLETE]
    np.random.shuffle(month_assignments)

    # Assign diagnoses
    diag_names = list(DIAGNOSES.keys())
    diag_probs = [DIAGNOSES[d]["freq"] for d in diag_names]
    diag_probs = [p / sum(diag_probs) for p in diag_probs]  # normalize
    assigned_diagnoses = np.random.choice(diag_names, size=N_COMPLETE, p=diag_probs)

    for i in range(N_COMPLETE):
        ip_no = ip_start + i
        month_start, month_end = month_assignments[i]
        diagnosis = assigned_diagnoses[i]
        diag_params = DIAGNOSES[diagnosis]

        # Date of admission (random within assigned month)
        admission_date = random_date_in_range(month_start, month_end, 1)[0]

        # Date of birth (usually same day or 0-3 days before admission)
        birth_delay = np.random.choice([0, 0, 0, 0, 1, 1, 2, 3, 5, 10])
        birth_date = admission_date - timedelta(days=int(birth_delay))

        # Birth weight (from diagnosis-specific distribution, clipped)
        bw = np.random.normal(diag_params["bw_mean"], diag_params["bw_std"])
        bw = np.clip(bw, 0.55, 5.5)
        bw = round(bw, 2)

        # Sex
        sex = np.random.choice(["Male", "Female"], p=[0.571, 0.429])

        # Mother's age
        age = int(np.clip(np.random.normal(25.5, 6.4), 15, 48))

        # HIV status
        hiv = np.random.choice(["TR", "Unknown", "TRR"], p=[0.52, 0.46, 0.02])

        # Mode of delivery
        mode = np.random.choice(
            ["Spontaneous Vaginal Delivery", "Caesarean Section"],
            p=[0.69, 0.31]
        )

        # How many babies
        plurality = np.random.choice(
            ["Singleton", "Twin", "Triplet"],
            p=[0.85, 0.13, 0.02]
        )
        # Adjust weight for multiples
        if plurality == "Twin":
            bw = np.clip(bw * 0.75, 0.55, 4.0)
            bw = round(bw, 2)
        elif plurality == "Triplet":
            bw = np.clip(bw * 0.60, 0.55, 3.5)
            bw = round(bw, 2)

        # Place of birth / referral (correlated)
        place = np.random.choice(ALL_FACILITIES)
        referral = place  # usually same

        # District
        district = np.random.choice(district_names, p=district_probs)

        # Village
        village = np.random.choice(VILLAGES)

        # Reason for admission
        reason = np.random.choice(REASONS.get(diagnosis, ["Unknown"]))

        # Outcome assignment (mutually exclusive: 7-day OR 28-day OR neither)
        outcome_phase = np.random.choice(["7d", "28d", "none"], p=[0.35, 0.35, 0.30])

        s7_outcome, s7_date = "", ""
        s28_outcome, s28_date = "", ""

        if outcome_phase == "7d":
            # 7-day outcome
            died = np.random.random() < diag_params["mortality_7d"]
            if died:
                s7_outcome = "DD"
            else:
                s7_outcome = np.random.choice(["D", "S", "R"], p=[0.82, 0.15, 0.03])
            days_to_outcome = int(np.clip(np.random.normal(3.4, 2.0), 0, 7))
            s7_date = (admission_date + timedelta(days=days_to_outcome)).strftime("%d.%m.%y")
        elif outcome_phase == "28d":
            # 28-day outcome
            died = np.random.random() < diag_params["mortality_28d"]
            if died:
                s28_outcome = "DD"
            else:
                s28_outcome = np.random.choice(["D", "S", "R"], p=[0.88, 0.08, 0.04])
            days_to_outcome = int(np.clip(np.random.normal(8.4, 4.0), 1, 28))
            s28_date = (admission_date + timedelta(days=days_to_outcome)).strftime("%d.%m.%y")

        rows.append({
            "IP No.": ip_no,
            "Address (District)": district,
            "Address (Village)": village,
            "Mother's Age in years": age,
            "HIV Status Code": hiv,
            "Sex": sex,
            "Date of Birth": birth_date.strftime("%d.%m.%y"),
            "Time of Birth": random_time(),
            "Place of Birth": place,
            "Mode of Delivery": mode,
            "Birth Weight (Kg)": bw,
            "How many babies born": plurality,
            "Source of warmth": "",
            "BCG ": "",
            "Polio": "",
            "Date of Admission": admission_date.strftime("%d.%m.%y"),
            "Time of Admission": random_time(),
            "Reason for Admission": reason,
            "Blood Transfusion": "",
            "Final Diagnosis": diagnosis,
            "Other Diagnosis": "",
            "Referral From": referral,
            "Discharge Weight (kg)": "",
            "Status at 7 Days - Outcome": s7_outcome,
            "Status at 7 Days - Date": s7_date,
            "Status at 28 Days - Outcome": s28_outcome,
            "Status at 28 Days - Date": s28_date,
            "Service Provider": "",
        })

    # Add skeleton rows (empty except IP No.)
    for i in range(N_SKELETON):
        ip_no = ip_start + N_COMPLETE + i
        rows.append({
            "IP No.": ip_no,
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

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/Logbook.csv", index=False, encoding="latin-1")
    print(f"Generated {len(df)} synthetic records")
    print(f"  Complete rows: {N_COMPLETE}")
    print(f"  Skeleton rows: {N_SKELETON}")

    # Quick validation
    complete = df[df["Date of Admission"] != ""]
    print(f"  Admissions with dates: {len(complete)}")
    deaths_7d = (complete["Status at 7 Days - Outcome"] == "DD").sum()
    deaths_28d = (complete["Status at 28 Days - Outcome"] == "DD").sum()
    print(f"  Deaths (7d): {deaths_7d}, Deaths (28d): {deaths_28d}")
    print(f"  Total deaths: {deaths_7d + deaths_28d}")
    print(f"  Mortality: {(deaths_7d + deaths_28d) / len(complete) * 100:.1f}%")
