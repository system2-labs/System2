#!/usr/bin/env python3
"""generate_med_grounded_dataset.py - High-Purity Grounded Medical Reasoning Generator.

Lifts Track C from raw academic trivia to 3,500+ pure grounded clinical diagnostic
and pharmacokinetic calculation samples. Implements the In-Domain Causal Purity Law
and Type-Aware Distractor Hardening:
1. Arterial Blood Gas (ABG) & Acid-Base Diagnostics (Winter's formula, anion gap, compensation)
2. Cardiovascular Hemodynamics & Shock Profiles (Hypovolemic, Cardiogenic, Septic, Neurogenic)
3. Clinical Pharmacokinetics & Drug Dosing (half-life, clearance, loading dose, maintenance dose)
4. Renal Pathophysiology & Electrolyte Imbalance (FeNa prerenal vs ATN, sodium correction rate)
5. Pulmonary Function Tests (FEV1/FVC obstruction vs restriction, DLCO diffusion capacity)
6. Endocrine Feedback Axes (Thyroid TSH/FT4, Adrenal Cortisol/ACTH diagnostic pathways)
7. Clinical Fluid Resuscitation & Formulas (Parkland burn formula, Holliday-Segar 4-2-1 rule)

Every distractor models an authentic clinical procedural error with character-length parity.
Output: ./data/med_grounded_expanded (train and test splits)
"""

import math
import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_from_disk

OUTPUT_DIR = "./data/med_grounded_expanded"
TRACK_C_EXPANDED_DIR = "./data/med_expanded"


def build_sample(q, correct, distractors, source, rng):
    """Format 4 choices with length balance and randomized positions."""
    clean_distractors = []
    for d in distractors:
        if d != correct and d not in clean_distractors:
            clean_distractors.append(d)

    if len(clean_distractors) < 3:
        return None

    selected = clean_distractors[:3]
    choices = [correct] + selected

    max_len = max(len(c) for c in choices)
    norm_choices = []
    for c in choices:
        if len(c) < max_len * 0.82:
            c = f"{c} (standard clinical guideline)"
        norm_choices.append(c)

    c_tuples = list(enumerate(norm_choices))
    rng.shuffle(c_tuples)
    shuffled = [t[1] for t in c_tuples]
    label = [t[0] for t in c_tuples].index(0)

    formatted_text = (
        f"Question: {q}\n"
        f"(A) {shuffled[0]}\n"
        f"(B) {shuffled[1]}\n"
        f"(C) {shuffled[2]}\n"
        f"(D) {shuffled[3]}"
    )

    return {
        "question": q,
        "choices": shuffled,
        "text": formatted_text,
        "label": label,
        "source": source,
    }


def generate_abg_acid_base(rng, n=600):
    """Generate ABG and Acid-Base calculation and diagnostic questions."""
    samples = []
    cases = [
        # (type, pH_range, pco2_range, hco3_range, desc)
        ("met_acid_high_ag", (7.18, 7.32), (24, 34), (10, 18), "Metabolic acidosis with elevated anion gap"),
        ("met_acid_norm_ag", (7.22, 7.33), (26, 35), (12, 19), "Metabolic acidosis with normal anion gap (hyperchloremic)"),
        ("met_alk", (7.48, 7.56), (42, 52), (30, 38), "Metabolic alkalosis with compensatory respiratory hypoventilation"),
        ("resp_acid_acute", (7.24, 7.33), (52, 68), (24, 27), "Acute uncompensated respiratory acidosis"),
        ("resp_acid_chronic", (7.34, 7.39), (58, 74), (32, 40), "Chronic compensated respiratory acidosis (renal retention of bicarbonate)"),
        ("resp_alk_acute", (7.50, 7.58), (22, 30), (22, 25), "Acute uncompensated respiratory alkalosis from hyperventilation"),
    ]

    for _ in range(n):
        c_type, ph_r, pco2_r, hco3_r, diagnosis = rng.choice(cases)
        ph = round(rng.uniform(*ph_r), 2)
        pco2 = rng.randint(*pco2_r)
        hco3 = rng.randint(*hco3_r)
        na = rng.randint(136, 144)
        cl = rng.randint(96, 108)

        ag = na - (cl + hco3)
        expected_pco2 = round(1.5 * hco3 + 8, 1)

        q = (
            f"An arterial blood gas (ABG) and basic metabolic panel reveal: "
            f"pH {ph:.2f}, PaCO2 {pco2} mmHg, HCO3- {hco3} mEq/L, Na+ {na} mEq/L, and Cl- {cl} mEq/L. "
            f"What is the primary acid-base diagnosis and compensatory state?"
        )

        correct = f"{diagnosis}; Anion Gap is {ag} mEq/L (expected PaCO2 is {expected_pco2} +- 2 mmHg)"

        distractors = [
            f"Metabolic alkalosis; Anion Gap is {ag + 6} mEq/L with complete respiratory compensation",
            f"Acute respiratory alkalosis; Anion Gap is {max(4, ag - 6)} mEq/L with secondary renal wasting",
            f"Normal acid-base status with spurious laboratory artifact; Anion Gap is 12 mEq/L",
            f"Mixed respiratory and metabolic alkalosis without appropriate renal compensation",
        ]

        sample = build_sample(q, correct, distractors, "med_abg_acid_base", rng)
        if sample:
            samples.append(sample)

    return samples


def generate_hemodynamics_shock(rng, n=600):
    """Generate Cardiovascular Hemodynamics & Shock Classification questions."""
    samples = []
    shock_types = [
        {
            "name": "Hypovolemic Shock",
            "cvp": (1, 4), "ci": (1.4, 2.1), "svr": (1600, 2400), "svo2": (48, 62),
            "etiology": "severe volume depletion or acute hemorrhage requiring rapid crystalloid/blood resuscitation",
            "wrong": ["Cardiogenic Shock", "Septic / Distributive Shock", "Neurogenic Shock"],
        },
        {
            "name": "Cardiogenic Shock",
            "cvp": (12, 20), "ci": (1.2, 1.9), "svr": (1800, 2600), "svo2": (45, 58),
            "etiology": "primary pump failure requiring inotropic support (e.g., dobutamine) and afterload reduction",
            "wrong": ["Hypovolemic Shock", "Septic / Distributive Shock", "Anaphylactic Shock"],
        },
        {
            "name": "Septic (Distributive) Shock",
            "cvp": (3, 8), "ci": (3.8, 5.5), "svr": (450, 750), "svo2": (76, 88),
            "etiology": "profound vasodilation and decreased systemic vascular resistance requiring norepinephrine",
            "wrong": ["Hypovolemic Shock", "Cardiogenic Shock", "Obstructive Shock"],
        },
        {
            "name": "Neurogenic Shock",
            "cvp": (2, 5), "ci": (1.8, 2.4), "svr": (500, 800), "svo2": (55, 68),
            "etiology": "loss of sympathetic vascular tone and paradoxical bradycardia following spinal cord injury",
            "wrong": ["Hypovolemic Shock", "Septic / Distributive Shock", "Cardiogenic Shock"],
        },
    ]

    for _ in range(n):
        st = rng.choice(shock_types)
        map_val = rng.randint(50, 62)
        hr = rng.randint(45, 55) if st["name"] == "Neurogenic Shock" else rng.randint(110, 138)
        cvp = rng.randint(*st["cvp"])
        ci = round(rng.uniform(*st["ci"]), 2)
        svr = rng.randint(*st["svr"])
        svo2 = rng.randint(*st["svo2"])

        q = (
            f"A patient in the intensive care unit presents with MAP {map_val} mmHg, HR {hr} bpm, "
            f"Central Venous Pressure (CVP) {cvp} mmHg (normal 2-8), Cardiac Index (CI) {ci:.2f} L/min/m2 (normal 2.5-4.0), "
            f"Systemic Vascular Resistance (SVR) {svr} dynes-sec/cm5 (normal 800-1200), and ScvO2 {svo2}%. "
            f"Which shock category and hemodynamic profile does this represent?"
        )

        correct = f"{st['name']} characterized by {st['etiology']}"

        distractors = [
            f"{st['wrong'][0]} characterized by elevated pulmonary capillary wedge pressure and reflex bradycardia",
            f"{st['wrong'][1]} characterized by hyperdynamic cardiac index and refractory peripheral vasoconstriction",
            f"{st['wrong'][2]} characterized by primary cardiac tamponade with equalized diastolic chamber pressures",
        ]

        sample = build_sample(q, correct, distractors, "med_hemodynamics_shock", rng)
        if sample:
            samples.append(sample)

    return samples


def generate_pharmacokinetics(rng, n=550):
    """Generate Clinical Pharmacokinetics & Drug Dosing questions."""
    samples = []

    for _ in range(n):
        mode = rng.choice(["half_life", "loading_dose", "maintenance_dose", "steady_state"])
        vd = rng.randint(15, 80)      # L
        cl = round(rng.uniform(1.2, 6.5), 1)  # L/h
        t_half = round((0.693 * vd) / cl, 1)

        if mode == "half_life":
            q = (
                f"A drug with first-order linear elimination has a volume of distribution (Vd) of {vd} L "
                f"and a systemic clearance (CL) of {cl} L/h. What is the elimination half-life (t1/2) of this drug?"
            )
            correct = f"{t_half} hours (calculated via t1/2 = 0.693 * Vd / CL)"
            distractors = [
                f"{round(vd / cl, 1)} hours (calculated without logarithmic conversion factor)",
                f"{round((cl * vd) / 10.0, 1)} hours (calculated assuming zero-order clearance saturation)",
                f"{round(t_half * 2.5, 1)} hours (calculated assuming tissue accumulation equilibrium)",
            ]
        elif mode == "loading_dose":
            target_cp = rng.randint(10, 30)  # mg/L
            ld = int(round(target_cp * vd))
            q = (
                f"A clinician desires an immediate target plasma concentration (Cp) of {target_cp} mg/L for an IV medication. "
                f"The patient's apparent volume of distribution (Vd) is {vd} L. What is the required IV loading dose?"
            )
            correct = f"{ld} mg (calculated via Loading Dose = Target Cp * Vd)"
            distractors = [
                f"{int(round(ld * 0.5))} mg (calculated assuming 50% bioavailability for IV administration)",
                f"{int(round(target_cp * cl * 10))} mg (calculated using clearance instead of volume of distribution)",
                f"{int(round(ld * 1.8))} mg (calculated with unadjusted extracellular fluid overcompensation)",
            ]
        elif mode == "maintenance_dose":
            target_css = rng.randint(5, 25)  # mg/L
            dosing_interval = rng.choice([6, 8, 12, 24])  # hours
            md_rate = round(target_css * cl, 1)  # mg/h
            md_total = int(round(md_rate * dosing_interval))
            q = (
                f"To maintain a steady-state concentration (Css) of {target_css} mg/L given every {dosing_interval} hours (IV), "
                f"with systemic clearance (CL) of {cl} L/h, what is the appropriate maintenance dose per interval?"
            )
            correct = f"{md_total} mg every {dosing_interval} hours (calculated via MD = Css * CL * interval)"
            distractors = [
                f"{int(round(md_total * 0.5))} mg every {dosing_interval} hours (underestimating clearance rate)",
                f"{int(round(target_css * vd))} mg every {dosing_interval} hours (substituting Vd for clearance rate)",
                f"{int(round(md_total * 1.7))} mg every {dosing_interval} hours (incorrectly adding loading dose fraction)",
            ]
        else:  # steady_state
            ss_time_4 = round(4 * t_half, 1)
            ss_time_5 = round(5 * t_half, 1)
            q = (
                f"A continuous IV infusion of a drug is initiated. If the drug has an elimination half-life of {t_half} hours, "
                f"approximately how long will it take to reach therapeutic steady-state concentration (>93-97%)?"
            )
            correct = f"{ss_time_4} to {ss_time_5} hours (equivalent to 4 to 5 elimination half-lives)"
            distractors = [
                f"{round(t_half, 1)} hours (equivalent to 1 half-life where only 50% steady-state is reached)",
                f"{round(2 * t_half, 1)} hours (equivalent to 2 half-lives where only 75% steady-state is reached)",
                f"{round(10 * t_half, 1)} hours (unnecessarily delaying assessment beyond steady-state equilibrium)",
            ]

        sample = build_sample(q, correct, distractors, "med_pharmacokinetics", rng)
        if sample:
            samples.append(sample)

    return samples


def generate_renal_fena(rng, n=550):
    """Generate Renal Function, FeNa & Electrolyte calculation questions."""
    samples = []

    for _ in range(n):
        mode = rng.choice(["fena_calc", "sodium_correction", "hyperkalemia_ecg"])

        if mode == "fena_calc":
            is_prerenal = rng.choice([True, False])
            if is_prerenal:
                una = rng.randint(8, 18)
                pna = rng.randint(136, 142)
                pcr = round(rng.uniform(1.8, 3.2), 1)
                ucr = rng.randint(80, 160)
                bun = int(round(pcr * rng.randint(22, 35)))
                fena = round((una * pcr) / (pna * ucr) * 100, 2)
                diagnosis = "Prerenal azotemia"
                mech = "intact tubular sodium reabsorption in response to renal hypoperfusion (FeNa < 1% and BUN/Cr > 20:1)"
                wrong_diag = "Acute Tubular Necrosis (ATN)"
            else:
                una = rng.randint(45, 85)
                pna = rng.randint(134, 140)
                pcr = round(rng.uniform(2.5, 4.5), 1)
                ucr = rng.randint(25, 45)
                bun = int(round(pcr * rng.randint(8, 14)))
                fena = round((una * pcr) / (pna * ucr) * 100, 2)
                diagnosis = "Intrinsic Acute Tubular Necrosis (ATN)"
                mech = "impaired tubular sodium reabsorption from tubular epithelial cell injury (FeNa > 2% and BUN/Cr < 15:1)"
                wrong_diag = "Prerenal azotemia"

            q = (
                f"A patient with acute kidney injury has serum creatinine {pcr} mg/dL, BUN {bun} mg/dL, serum Na+ {pna} mEq/L, "
                f"urine Na+ {una} mEq/L, and urine creatinine {ucr} mg/dL. "
                f"What is the Fractional Excretion of Sodium (FeNa) and the most likely etiology?"
            )
            correct = f"FeNa is {fena:.2f}%; consistent with {diagnosis} due to {mech}"
            distractors = [
                f"FeNa is {round(fena * 3.5, 2)}%; consistent with {wrong_diag} with secondary obstructive uropathy",
                f"FeNa is {round(max(0.1, fena * 0.25), 2)}%; consistent with post-renal obstructive nephropathy",
                f"FeNa calculation is invalid in acute kidney injury; patient requires urgent emergent dialysis",
            ]
        elif mode == "sodium_correction":
            current_na = rng.randint(114, 122)
            glucose = rng.randint(450, 850)
            correction = round(1.6 * ((glucose - 100) / 100.0), 1)
            corrected_na = round(current_na + correction, 1)

            q = (
                f"A patient with severe diabetic ketoacidosis has blood glucose {glucose} mg/dL and measured serum Na+ {current_na} mEq/L. "
                f"What is the corrected serum sodium concentration accounting for hyperglycemia-induced osmotic fluid shift?"
            )
            correct = f"{corrected_na} mEq/L (add 1.6 mEq/L Na+ per 100 mg/dL glucose elevation above 100 mg/dL)"
            distractors = [
                f"{current_na} mEq/L (measured serum sodium requires no correction in hyperglycemic states)",
                f"{round(current_na - correction, 1)} mEq/L (erroneously subtracting the osmotic dilution factor)",
                f"{round(current_na + 3.0 * ((glucose - 100) / 100.0), 1)} mEq/L (overcorrecting with arbitrary formula)",
            ]
        else:  # hyperkalemia_ecg
            k_val = round(rng.uniform(6.8, 8.2), 1)
            q = (
                f"A patient with end-stage renal disease presents with serum potassium of {k_val} mEq/L. "
                f"What is the immediate priority medical intervention to stabilize the cardiac myocyte membrane?"
            )
            correct = "Intravenous Calcium Gluconate (10%) to stabilize the cardiac membrane and prevent ventricular arrhythmias"
            distractors = [
                "Intravenous regular insulin and 50% dextrose as the sole first-line membrane-stabilizing therapy",
                "Sodium polystyrene sulfonate (Kayexalate) orally for immediate hyperacute cardiac protection",
                "Hemodialysis catheter placement before administering any pharmacologic stabilizing agents",
            ]

        sample = build_sample(q, correct, distractors, "med_renal_electrolytes", rng)
        if sample:
            samples.append(sample)

    return samples


def generate_pulmonary_pfts(rng, n=550):
    """Generate Pulmonary Function Tests (PFT) & Spirometry questions."""
    samples = []
    cases = [
        {
            "diagnosis": "Asthma (Reversible Obstructive Defect)",
            "fev1_fvc": 0.58, "dlco": "Normal (95% predicted)",
            "post_bd": "Significant bronchodilator reversibility (FEV1 increases by >12% and >200 mL)",
            "fev1": 56, "fvc": 88,
        },
        {
            "diagnosis": "Emphysema / COPD (Irreversible Obstructive with Impaired Diffusion)",
            "fev1_fvc": 0.52, "dlco": "Severely reduced (45% predicted due to alveolar-capillary destruction)",
            "post_bd": "Minimal bronchodilator reversibility (<12% and <200 mL increase in FEV1)",
            "fev1": 48, "fvc": 82,
        },
        {
            "diagnosis": "Idiopathic Pulmonary Fibrosis (Intrinsic Restrictive Defect with Low DLCO)",
            "fev1_fvc": 0.86, "dlco": "Reduced (50% predicted due to interstitial fibrosis)",
            "post_bd": "No change following bronchodilator administration",
            "fev1": 62, "fvc": 58,
        },
        {
            "diagnosis": "Extrinsic Chest Wall / Obesity Hypoventilation (Restrictive with Normal DLCO)",
            "fev1_fvc": 0.84, "dlco": "Preserved / Normal (92% predicted because lung parenchyma is intact)",
            "post_bd": "No change following bronchodilator administration",
            "fev1": 65, "fvc": 62,
        },
    ]

    for _ in range(n):
        c = rng.choice(cases)
        ratio = round(c["fev1_fvc"] + rng.uniform(-0.03, 0.03), 2)
        fev1 = c["fev1"] + rng.randint(-4, 4)
        fvc = c["fvc"] + rng.randint(-4, 4)

        q = (
            f"Pulmonary function tests (PFTs) in a dyspneic patient demonstrate: "
            f"FEV1 {fev1}% predicted, FVC {fvc}% predicted, FEV1/FVC ratio {ratio:.2f} (normal >= 0.70), "
            f"and DLCO {c['dlco']}. Post-bronchodilator testing shows: {c['post_bd']}. "
            f"What is the physiological interpretation?"
        )

        correct = f"{c['diagnosis']} (FEV1/FVC {'< 0.70 indicating obstruction' if ratio < 0.70 else '>= 0.70 with reduced lung volumes indicating restriction'})"

        wrong_diags = [x["diagnosis"] for x in cases if x["diagnosis"] != c["diagnosis"]]
        distractors = [
            f"{wrong_diags[0]} based on the observed DLCO and airflow characteristics",
            f"{wrong_diags[1]} with mixed ventilatory and parenchymal failure",
            f"Normal physiological lung function with effort-dependent artifactual variation",
        ]

        sample = build_sample(q, correct, distractors, "med_pulmonary_pfts", rng)
        if sample:
            samples.append(sample)

    return samples


def generate_endocrine_axes(rng, n=500):
    """Generate Endocrine Diagnostic Pathways & Feedback Loops questions."""
    samples = []
    cases = [
        ("Primary Hypothyroidism (Hashimoto)", "High TSH (>15 mIU/L), Low Free T4 (<0.6 ng/dL)", "failure of thyroid gland with loss of negative feedback on anterior pituitary"),
        ("Central / Secondary Hypothyroidism", "Low or inappropriately normal TSH (0.8 mIU/L), Low Free T4 (0.4 ng/dL)", "pituitary or hypothalamic insufficiency (requires MRI of pituitary before starting levothyroxine)"),
        ("Primary Hyperthyroidism (Graves Disease)", "Undetectable TSH (<0.01 mIU/L), Markedly elevated Free T4 (3.4 ng/dL)", "TSH-receptor stimulating antibodies driving autonomous thyroid hormone release"),
        ("Subclinical Hypothyroidism", "Mildly elevated TSH (7.5 mIU/L), Normal Free T4 (1.1 ng/dL)", "early compensatory thyroid state with preserved peripheral hormone concentrations"),
        ("Primary Adrenal Insufficiency (Addison)", "High morning ACTH (>120 pg/mL), Low morning Cortisol (<3 ug/dL)", "adrenocortical destruction with hyperpigmentation due to POMC/MSH co-secretion"),
        ("Secondary Adrenal Insufficiency", "Low morning ACTH (<5 pg/mL), Low morning Cortisol (<2 ug/dL)", "pituitary failure to secrete ACTH without cutaneous hyperpigmentation or mineralocorticoid loss"),
        ("Pituitary Cushing Disease", "Elevated morning Cortisol, Elevated morning ACTH, Suppresses with high-dose (8 mg) Dexamethasone", "ACTH-secreting pituitary microadenoma with partially preserved glucocorticoid negative feedback"),
        ("Ectopic ACTH Syndrome (Small Cell Lung CA)", "Severely elevated morning Cortisol, Markedly elevated ACTH, Fails to suppress with high-dose Dexamethasone", "autonomous paraneoplastic ACTH production bypassing pituitary glucocorticoid receptor regulation"),
    ]

    for _ in range(n):
        diagnosis, labs, mech = rng.choice(cases)

        q = (
            f"A patient evaluated for fatigue and endocrine dysfunction undergoes diagnostic hormone testing. "
            f"Results show: {labs}. "
            f"What is the most accurate diagnostic entity and underlying pathophysiological mechanism?"
        )

        correct = f"{diagnosis} caused by {mech}"

        other_cases = [x for x in cases if x[0] != diagnosis]
        rng.shuffle(other_cases)
        distractors = [
            f"{other_cases[0][0]} caused by {other_cases[0][2]}",
            f"{other_cases[1][0]} caused by peripheral hormone resistance syndromes",
            f"Euthyroid sick syndrome secondary to severe critical illness with transient deiodinase suppression",
        ]

        sample = build_sample(q, correct, distractors, "med_endocrine_axes", rng)
        if sample:
            samples.append(sample)

    return samples


def generate_fluid_formulas(rng, n=450):
    """Generate Parkland Burn & Pediatric Fluid Calculation questions."""
    samples = []

    for _ in range(n):
        mode = rng.choice(["parkland_burn", "holliday_segar"])

        if mode == "parkland_burn":
            weight = rng.randint(55, 95)  # kg
            tbsa = rng.randint(20, 55)   # %
            # Parkland: 4 mL * weight (kg) * % TBSA
            total_fluid = 4 * weight * tbsa
            first_8h = int(total_fluid / 2)
            rate_first_8h = int(round(first_8h / 8.0))
            second_16h = int(total_fluid / 2)
            rate_second_16h = int(round(second_16h / 16.0))

            q = (
                f"An adult patient weighing {weight} kg sustains second- and third-degree flame burns covering {tbsa}% of total body surface area (TBSA). "
                f"According to the Parkland formula, what is the total Lactated Ringer's resuscitation volume required in the first 24 hours, "
                f"and what is the infusion rate for the first 8 hours post-injury?"
            )
            correct = f"{total_fluid} mL total in 24 hours ({rate_first_8h} mL/h for the first 8 hours; {rate_second_16h} mL/h for the next 16 hours)"
            distractors = [
                f"{int(total_fluid * 0.5)} mL total in 24 hours ({int(rate_first_8h * 0.5)} mL/h continuous rate over 24 hours)",
                f"{int(total_fluid * 1.5)} mL total in 24 hours ({int(rate_first_8h * 1.5)} mL/h to prevent acute hypovolemic collapse)",
                f"{total_fluid} mL administered as an equal constant infusion of {int(total_fluid / 24.0)} mL/h across all 24 hours",
            ]
        else:  # holliday_segar
            weight = rng.randint(12, 35)  # kg
            # 4-2-1 rule:
            # First 10 kg: 4 mL/kg/h = 40 mL/h
            # Next 10 kg (11-20): 2 mL/kg/h
            # Remaining kg (>20): 1 mL/kg/h
            if weight <= 10:
                hourly = weight * 4
            elif weight <= 20:
                hourly = 40 + (weight - 10) * 2
            else:
                hourly = 40 + 20 + (weight - 20) * 1

            daily = hourly * 24

            q = (
                f"A child weighing {weight} kg is admitted for dehydration and placed on NPO status. "
                f"Using the standard Holliday-Segar (4-2-1 rule) method, what is the appropriate hourly maintenance IV fluid rate?"
            )
            correct = f"{hourly} mL/h ({daily} mL per 24 hours calculated via 4-2-1 rule)"
            distractors = [
                f"{int(weight * 4)} mL/h ({int(weight * 4 * 24)} mL/24h calculated applying 4 mL/kg across entire body weight)",
                f"{int(weight * 2)} mL/h ({int(weight * 2 * 24)} mL/24h underdosing maintenance fluid requirements)",
                f"{int(hourly * 1.6)} mL/h ({int(daily * 1.6)} mL/24h causing fluid overload and dilutional hyponatremia)",
            ]

        sample = build_sample(q, correct, distractors, "med_fluid_resuscitation", rng)
        if sample:
            samples.append(sample)

    return samples


def main():
    print("================================================================")
    print(" SYSTEM 2 GROUNDED CLINICAL DATASET GENERATOR (TRACK C)")
    print("================================================================")

    rng = random.Random(42)

    print("Generating grounded clinical reasoning domains...")
    samples = []
    samples.extend(generate_abg_acid_base(rng, n=650))
    samples.extend(generate_hemodynamics_shock(rng, n=650))
    samples.extend(generate_pharmacokinetics(rng, n=600))
    samples.extend(generate_renal_fena(rng, n=600))
    samples.extend(generate_pulmonary_pfts(rng, n=600))
    samples.extend(generate_endocrine_axes(rng, n=550))
    samples.extend(generate_fluid_formulas(rng, n=500))

    print(f"\nGenerated {len(samples)} grounded clinical calculation samples:")
    counts = Counter(s["source"] for s in samples)
    for src, c in counts.most_common():
        print(f"  - {src:30s}: {c:5d}")

    # Load high-quality clean USMLE / MedMCQA questions to maintain clinical breadths
    clean_vignettes = []
    if os.path.exists("./data/medical"):
        try:
            med_ds = load_from_disk("./data/medical")
            train_v = med_ds["train"]
            # Subsample 1,000 clean USMLE vignettes
            indices = list(range(len(train_v)))
            rng.shuffle(indices)
            for idx in indices[:1000]:
                ex = train_v[idx]
                clean_vignettes.append({
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "text": ex["text"],
                    "label": ex["label"],
                    "source": "usmle_clinical_vignette",
                })
            print(f"Ingested {len(clean_vignettes)} clean USMLE clinical vignettes.")
        except Exception as e:
            print(f"Warning: could not load data/medical: {e}")

    all_data = samples + clean_vignettes
    rng.shuffle(all_data)

    test_size = int(len(all_data) * 0.15)
    train_size = len(all_data) - test_size

    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    print(f"\nTotal Dataset: {len(all_data)} samples")
    print(f"Train split:   {len(train_data)} samples")
    print(f"Test split:    {len(test_data)} samples")

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_dict.save_to_disk(OUTPUT_DIR)
    print(f">>> Successfully saved grounded medical dataset to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
