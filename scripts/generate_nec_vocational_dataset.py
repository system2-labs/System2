#!/usr/bin/env python3
"""generate_nec_vocational_dataset.py - High-Purity Grounded NEC & Industrial Electrical Generator.

Lifts Track D from 522 to 3,000+ pure vocational samples.
Implements the In-Domain Causal Purity Law and Type-Aware Distractor Hardening:
1. NEC 210.19(A)(1) & NEC 215.2: Continuous loading & minimum OCPD / conductor ratings
2. NEC 310.16 Ampacity & Temperature Columns (60°C, 75°C, 90°C) with ambient & bundling deratings
3. NEC Chapter 9 Table 1: Conduit fill percentage calculations (1 conductor 53%, 2 conductors 31%, 3+ 40%)
4. NEC 314.16: Box fill volume allowances (conductors, clamps, devices 2x, grounds)
5. NEC 430.52: Motor branch circuit short-circuit & ground-fault protection (250% breaker, 175% fuse)
6. NEC 430.32: Motor overload protection sizing (125% vs 115% SF)
7. Single-phase and 3-phase Voltage Drop calculations (2KIL/CM and sqrt(3)KIL/CM)
8. NEC 250.66: Grounding Electrode Conductor (GEC) sizing
9. Multimeter & Megohmmeter Troubleshooting Fault Trees across industrial/commercial facilities

Every distractor represents an authentic procedural mistake with length parity across all 4 choices.
Output: ./data/electrical_expanded (and mirrored to track-d worktree)
"""

import math
import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_from_disk

OUTPUT_DIR = "./data/electrical_expanded"
TRACK_D_PURE_DIR = "./scratch/worktrees/track-d/data/electrical_pure"

STD_BREAKERS = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 600]

NEC_310_16_COPPER = {
    # AWG/kcmil: (60C, 75C, 90C)
    "14 AWG": (15, 20, 25),
    "12 AWG": (20, 25, 30),
    "10 AWG": (30, 35, 40),
    "8 AWG": (40, 50, 55),
    "6 AWG": (55, 65, 75),
    "4 AWG": (70, 85, 95),
    "3 AWG": (85, 100, 115),
    "2 AWG": (95, 115, 130),
    "1 AWG": (110, 130, 145),
    "1/0 AWG": (125, 150, 170),
    "2/0 AWG": (145, 175, 195),
    "3/0 AWG": (165, 200, 225),
    "4/0 AWG": (195, 230, 260),
    "250 kcmil": (215, 255, 290),
    "300 kcmil": (240, 285, 320),
    "350 kcmil": (260, 310, 350),
    "400 kcmil": (280, 335, 380),
    "500 kcmil": (320, 380, 430),
    "600 kcmil": (355, 420, 475),
}

BOX_FILL_VOLUMES = {
    "18 AWG": 1.50,
    "16 AWG": 1.75,
    "14 AWG": 2.00,
    "12 AWG": 2.25,
    "10 AWG": 2.50,
    "8 AWG": 3.00,
    "6 AWG": 5.00,
}

CIRCULAR_MILS = {
    "14 AWG": 4110,
    "12 AWG": 6530,
    "10 AWG": 10380,
    "8 AWG": 16510,
    "6 AWG": 26240,
    "4 AWG": 41740,
    "3 AWG": 52620,
    "2 AWG": 66360,
    "1 AWG": 83690,
    "1/0 AWG": 105600,
    "2/0 AWG": 133100,
    "3/0 AWG": 167800,
    "4/0 AWG": 211600,
    "250 kcmil": 250000,
    "300 kcmil": 300000,
    "350 kcmil": 350000,
    "500 kcmil": 500000,
}


def round_next_std_breaker(current):
    """Next standard overcurrent protection device rating."""
    for b in STD_BREAKERS:
        if b >= current:
            return b
    return int(math.ceil(current / 50.0) * 50)


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
            c = f"{c} (standard electrical specification)"
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


def generate_nec_continuous_loads(rng, n=750):
    """Generate NEC 210.19(A)(1) Continuous Load calculation questions."""
    samples = []
    currents = list(range(10, 200, 1)) + list(range(200, 480, 5))
    load_types = [
        "continuous commercial LED high-bay lighting",
        "continuous server rack power distribution unit",
        "continuous industrial duct electric heating",
        "continuous commercial store showcase luminaire",
        "continuous architectural exterior facade illumination",
        "continuous manufacturing process heat tape circuit",
    ]
    voltages = [120, 208, 240, 277, 480]

    for _ in range(n):
        load_a = rng.choice(currents)
        load_type = rng.choice(load_types)
        v = rng.choice(voltages)

        req_min = load_a * 1.25
        std_bkr = round_next_std_breaker(req_min)

        q = f"Under NEC 210.19(A)(1) and 210.20(A), what is the minimum overcurrent protective device rating required for a {load_type} load drawing {load_a} Amps continuously on a {v}V branch circuit?"

        correct = f"{std_bkr} Amps (calculated as {load_a}A multiplied by 125 percent continuous duty factor)"
        d1 = f"{load_a} Amps (calculated directly from steady-state continuous nameplate rating without multiplier)"
        d2 = f"{round_next_std_breaker(load_a * 1.50)} Amps (calculated as {load_a}A multiplied by 150 percent inductive appliance factor)"
        d3 = f"{round_next_std_breaker(load_a * 1.15)} Amps (calculated as {load_a}A multiplied by 115 percent motor overload factor)"

        s = build_sample(q, correct, [d1, d2, d3], "nec_continuous_load", rng)
        if s:
            samples.append(s)
    return samples


def generate_nec_ampacity_derating(rng, n=850):
    """Generate NEC 310.16 Ampacity & Bundling / Ambient Derating questions."""
    samples = []
    conductors = list(NEC_310_16_COPPER.keys())
    bundling_options = [
        (4, 6, 0.80, "80 percent bundling adjustment factor for 4 to 6 current-carrying conductors"),
        (7, 9, 0.70, "70 percent bundling adjustment factor for 7 to 9 current-carrying conductors"),
        (10, 20, 0.50, "50 percent bundling adjustment factor for 10 to 20 current-carrying conductors"),
        (21, 30, 0.45, "45 percent bundling adjustment factor for 21 to 30 current-carrying conductors"),
        (31, 40, 0.40, "40 percent bundling adjustment factor for 31 to 40 current-carrying conductors"),
    ]
    ambient_temps = [
        ("31°C–35°C (88°F–95°F)", 0.96, "0.96 ambient temperature correction factor"),
        ("36°C–40°C (97°F–104°F)", 0.91, "0.91 ambient temperature correction factor"),
        ("41°C–45°C (106°F–113°F)", 0.87, "0.87 ambient temperature correction factor"),
        ("46°C–50°C (115°F–122°F)", 0.82, "0.82 ambient temperature correction factor"),
    ]

    for _ in range(n):
        wire = rng.choice(conductors)
        amps_60, amps_75, amps_90 = NEC_310_16_COPPER[wire]
        mode = rng.choice(["bundling", "ambient", "combined"])

        if mode == "bundling":
            c_min, c_max, factor, factor_str = rng.choice(bundling_options)
            cond_count = rng.randint(c_min, c_max)
            derated_90 = round(amps_90 * factor, 1)
            final_amp = min(amps_75, derated_90)

            q = f"According to NEC 310.16 and 310.15(C)(1), what is the allowable derated ampacity of {wire} copper THHN conductors with 75°C terminations when {cond_count} current-carrying conductors are installed in a single raceway at 30°C ambient?"
            correct = f"{final_amp} Amps (calculated by applying the {int(factor*100)} percent adjustment factor to the 90°C base rating of {amps_90}A)"
            d1 = f"{amps_75} Amps (using unadjusted 75°C terminal rating without applying raceway mutual heating derating)"
            d2 = f"{round(amps_60 * factor, 1)} Amps (calculated by erroneously applying the {int(factor*100)} percent adjustment factor to 60°C column)"
            d3 = f"{amps_90} Amps (using unadjusted 90°C insulation rating without applying raceway mutual heating derating)"

        elif mode == "ambient":
            temp_desc, amb_factor, amb_str = rng.choice(ambient_temps)
            derated_90 = round(amps_90 * amb_factor, 1)
            final_amp = min(amps_75, derated_90)

            q = f"Under NEC 310.16 and Table 310.15(B)(1), what is the allowable ampacity of {wire} copper THHN conductors (75°C terminal rating) installed in an ambient temperature of {temp_desc}?"
            correct = f"{final_amp} Amps (applying the {amb_factor} ambient correction factor to the 90°C rating of {amps_90}A)"
            d1 = f"{amps_75} Amps (using the unadjusted 75°C terminal rating without ambient temperature correction)"
            d2 = f"{round(amps_60 * amb_factor, 1)} Amps (applying the ambient correction factor to the 60°C conductor column)"
            d3 = f"{amps_90} Amps (using the unadjusted 90°C rating without ambient temperature correction)"

        else:  # combined
            c_min, c_max, b_factor, _ = rng.choice(bundling_options)
            temp_desc, amb_factor, _ = rng.choice(ambient_temps)
            cond_count = rng.randint(c_min, c_max)
            tot_factor = round(b_factor * amb_factor, 3)
            derated_90 = round(amps_90 * tot_factor, 1)
            final_amp = min(amps_75, derated_90)

            q = f"Under NEC 310.16, what is the allowable derated ampacity of {wire} copper THHN conductors (75°C terminations) with {cond_count} current-carrying conductors in an ambient temperature of {temp_desc}?"
            correct = f"{final_amp} Amps (combining {int(b_factor*100)}% bundling and {amb_factor} ambient factors applied to {amps_90}A)"
            d1 = f"{amps_75} Amps (relying on standard 75°C termination ampacity without applying combined derating)"
            d2 = f"{round(amps_90 * b_factor, 1)} Amps (applying only raceway bundling derating while omitting ambient temperature)"
            d3 = f"{round(amps_90 * amb_factor, 1)} Amps (applying only ambient temperature correction while omitting raceway bundling)"

        s = build_sample(q, correct, [d1, d2, d3], "nec_ampacity_bundling", rng)
        if s:
            samples.append(s)
    return samples


def generate_nec_conduit_fill(rng, n=550):
    """Generate NEC Chapter 9 Table 1 Conduit Fill questions."""
    samples = []
    scenarios = [
        (1, "53 percent maximum cross-sectional raceway fill", "31 percent maximum cross-sectional raceway fill", "40 percent maximum cross-sectional raceway fill", "60 percent maximum cross-sectional raceway fill"),
        (2, "31 percent maximum cross-sectional raceway fill", "40 percent maximum cross-sectional raceway fill", "53 percent maximum cross-sectional raceway fill", "25 percent maximum cross-sectional raceway fill"),
        (3, "40 percent maximum cross-sectional raceway fill", "31 percent maximum cross-sectional raceway fill", "53 percent maximum cross-sectional raceway fill", "45 percent maximum cross-sectional raceway fill"),
        (4, "40 percent maximum cross-sectional raceway fill", "50 percent maximum cross-sectional raceway fill", "31 percent maximum cross-sectional raceway fill", "35 percent maximum cross-sectional raceway fill"),
        (5, "40 percent maximum cross-sectional raceway fill", "31 percent maximum cross-sectional raceway fill", "45 percent maximum cross-sectional raceway fill", "50 percent maximum cross-sectional raceway fill"),
        (6, "40 percent maximum cross-sectional raceway fill", "45 percent maximum cross-sectional raceway fill", "31 percent maximum cross-sectional raceway fill", "50 percent maximum cross-sectional raceway fill"),
        (8, "40 percent maximum cross-sectional raceway fill", "31 percent maximum cross-sectional raceway fill", "53 percent maximum cross-sectional raceway fill", "35 percent maximum cross-sectional raceway fill"),
        (12, "40 percent maximum cross-sectional raceway fill", "50 percent maximum cross-sectional raceway fill", "31 percent maximum cross-sectional raceway fill", "45 percent maximum cross-sectional raceway fill"),
    ]

    raceways = [
        "Electrical Metallic Tubing (EMT)",
        "Rigid Metal Conduit (RMC)",
        "Schedule 40 PVC conduit",
        "Schedule 80 PVC conduit",
        "Intermediate Metal Conduit (IMC)",
        "Flexible Metal Conduit (FMC)",
    ]
    wires = ["14 AWG THHN", "12 AWG THHN", "10 AWG THHN", "8 AWG THHN", "6 AWG THHN", "4 AWG THHN", "2 AWG THHN", "1/0 AWG THHN"]

    for _ in range(n):
        num_cond, corr_pct, d1_pct, d2_pct, d3_pct = rng.choice(scenarios)
        raceway = rng.choice(raceways)
        wire = rng.choice(wires)

        q = f"Under NEC Chapter 9, Table 1, what is the maximum permissible percentage conduit fill when installing {num_cond} {wire} conductors in {raceway}?"

        correct = f"{corr_pct} (mandated for {'one conductor' if num_cond == 1 else 'two conductors' if num_cond == 2 else 'three or more conductors'})"
        d1 = f"{d1_pct} (standard misapplication of the adjacent conductor count fill bracket)"
        d2 = f"{d2_pct} (standard misapplication of the single or double conductor fill bracket)"
        d3 = f"{d3_pct} (informational control circuit allowance not recognized under Chapter 9)"

        s = build_sample(q, correct, [d1, d2, d3], "nec_conduit_fill", rng)
        if s:
            samples.append(s)
    return samples


def generate_nec_box_fill(rng, n=650):
    """Generate NEC 314.16 Box Fill Volume calculation questions."""
    samples = []
    wire_sizes = list(BOX_FILL_VOLUMES.keys())

    for _ in range(n):
        wire = rng.choice(wire_sizes)
        unit_vol = BOX_FILL_VOLUMES[wire]

        num_wires = rng.randint(3, 14)
        num_clamps = rng.choice([0, 1])
        num_devices = rng.choice([1, 2, 3])
        num_grounds = rng.randint(1, 6)

        # Ground count rule: 1 allowance for up to 4 grounds, plus 0.25 allowance for each ground beyond 4
        ground_allowances = 1.0 + max(0, num_grounds - 4) * 0.25
        total_counts = num_wires + (1.0 if num_clamps > 0 else 0.0) + (num_devices * 2.0) + ground_allowances
        correct_vol = round(total_counts * unit_vol, 2)

        # Distractor 1: forgot device counts as 2x (counted as 1x)
        err1_counts = num_wires + (1.0 if num_clamps > 0 else 0.0) + (num_devices * 1.0) + ground_allowances
        d1_vol = round(err1_counts * unit_vol, 2)

        # Distractor 2: counted each ground as a full individual conductor
        err2_counts = num_wires + (1.0 if num_clamps > 0 else 0.0) + (num_devices * 2.0) + float(num_grounds)
        d2_vol = round(err2_counts * unit_vol, 2)

        # Distractor 3: omitted internal clamps and grounds completely
        err3_counts = num_wires + (num_devices * 2.0)
        d3_vol = round(err3_counts * unit_vol, 2)

        clamp_str = "with internal cable clamps" if num_clamps > 0 else "without internal cable clamps"
        device_str = "one duplex receptacle" if num_devices == 1 else f"{num_devices} yoke devices (receptacles/switches)"
        q = f"Under NEC 314.16(B), what is the minimum box volume required for a metal enclosure containing {num_wires} {wire} conductors, {clamp_str}, {device_str}, and {num_grounds} equipment grounding conductors?"

        correct = f"{correct_vol} cubic inches (based on {total_counts} volume allowances at {unit_vol} cu in per {wire} conductor)"
        d1 = f"{d1_vol} cubic inches (erroneously counting the yoke device as a single conductor volume allowance)"
        d2 = f"{d2_vol} cubic inches (erroneously allocating an individual volume allowance for every grounding conductor)"
        d3 = f"{d3_vol} cubic inches (omitting internal clamp and equipment grounding conductor volume allocations)"

        s = build_sample(q, correct, [d1, d2, d3], "nec_box_fill", rng)
        if s:
            samples.append(s)
    return samples


def generate_nec_motor_protection(rng, n=700):
    """Generate NEC 430.52 Motor Branch Short-Circuit & Overcurrent Protection questions."""
    samples = []
    flcs = [7.6, 9.6, 11, 14, 15.2, 17.5, 21, 22, 27, 28, 34, 40, 42, 52, 54, 65, 77, 96, 124, 156]
    voltages = [115, 208, 230, 460, 575]

    for _ in range(n):
        flc = rng.choice(flcs)
        v = rng.choice(voltages)
        hp = rng.choice([3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100])
        prot_type = rng.choice(["inverse_time_breaker", "dual_element_fuse", "non_time_delay_fuse"])

        if prot_type == "inverse_time_breaker":
            calc_val = flc * 2.50
            std_val = round_next_std_breaker(calc_val)
            q = f"For a {hp} HP, {v}V 3-phase squirrel cage induction motor with an NEC Table full-load current (FLC) of {flc} Amps, what is the maximum standard inverse-time circuit breaker rating for branch short-circuit protection under NEC 430.52?"
            correct = f"{std_val} Amps (calculated as {flc}A FLC multiplied by 250 percent rounded to next standard size)"
            d1 = f"{round_next_std_breaker(flc * 1.75)} Amps (calculated using the 175 percent multiplier for dual-element time-delay fuses)"
            d2 = f"{int(calc_val)} Amps (calculated as 250 percent without rounding up to the next standard breaker rating)"
            d3 = f"{round_next_std_breaker(flc * 1.25)} Amps (calculated using the 125 percent multiplier for running overload thermal sizing)"

        elif prot_type == "dual_element_fuse":
            calc_val = flc * 1.75
            std_val = round_next_std_breaker(calc_val)
            q = f"For a {hp} HP, {v}V 3-phase squirrel cage motor with a Table full-load current (FLC) of {flc} Amps, what is the maximum standard dual-element time-delay fuse rating under NEC 430.52?"
            correct = f"{std_val} Amps (calculated as {flc}A FLC multiplied by 175 percent rounded to next standard size)"
            d1 = f"{round_next_std_breaker(flc * 2.50)} Amps (calculated using the 250 percent multiplier for inverse-time circuit breakers)"
            d2 = f"{round_next_std_breaker(flc * 3.00)} Amps (calculated using the 300 percent multiplier for non-time-delay fuses)"
            d3 = f"{round_next_std_breaker(flc * 1.15)} Amps (calculated using the 115 percent motor thermal overload continuous factor)"

        else:  # non_time_delay_fuse
            calc_val = flc * 3.00
            std_val = round_next_std_breaker(calc_val)
            q = f"For a {hp} HP, {v}V 3-phase induction motor with a Table full-load current (FLC) of {flc} Amps, what is the maximum standard non-time-delay fuse rating permitted by NEC 430.52?"
            correct = f"{std_val} Amps (calculated as {flc}A FLC multiplied by 300 percent rounded to next standard size)"
            d1 = f"{round_next_std_breaker(flc * 1.75)} Amps (calculated using the 175 percent multiplier for dual-element time-delay fuses)"
            d2 = f"{round_next_std_breaker(flc * 2.50)} Amps (calculated using the 250 percent multiplier for inverse-time circuit breakers)"
            d3 = f"{round_next_std_breaker(flc * 1.25)} Amps (calculated using the 125 percent motor thermal overload continuous factor)"

        s = build_sample(q, correct, [d1, d2, d3], "nec_motor_protection", rng)
        if s:
            samples.append(s)
    return samples


def generate_voltage_drop(rng, n=750):
    """Generate Single-phase and 3-phase Voltage Drop questions."""
    samples = []
    wire_choices = list(CIRCULAR_MILS.keys())

    for _ in range(n):
        phase = rng.choice(["single-phase", "three-phase"])
        wire = rng.choice(wire_choices)
        cm = CIRCULAR_MILS[wire]
        current = rng.choice([15, 16, 20, 24, 30, 32, 40, 48, 50, 60, 70, 80, 100, 125, 150, 175, 200])
        length_ft = rng.choice([60, 80, 100, 120, 150, 180, 200, 240, 250, 300, 350, 400, 450, 500])
        k = 12.9  # copper

        if phase == "single-phase":
            vd = round((2 * k * current * length_ft) / cm, 2)
            formula_desc = "using the single-phase formula Vd = (2 * K * I * L) / CM"
            wrong_formula_vd = round((math.sqrt(3) * k * current * length_ft) / cm, 2)
            wrong_desc = "erroneously applying the three-phase square-root-of-three multiplier"
        else:
            vd = round((math.sqrt(3) * k * current * length_ft) / cm, 2)
            formula_desc = "using the three-phase formula Vd = (1.732 * K * I * L) / CM"
            wrong_formula_vd = round((2 * k * current * length_ft) / cm, 2)
            wrong_desc = "erroneously applying the single-phase multiplier of two"

        d2_vd = round(vd * 0.5, 2)
        d3_vd = round(vd * 1.5, 2)

        v_sys = 120 if phase == "single-phase" else rng.choice([208, 480])
        q = f"Calculate the approximate line-to-line voltage drop for a {phase} circuit operating at {v_sys}V carrying {current} Amps over a one-way distance of {length_ft} feet using {wire} copper conductors (K = 12.9)."

        correct = f"{vd} Volts drop ({formula_desc})"
        d1 = f"{wrong_formula_vd} Volts drop ({wrong_desc})"
        d2 = f"{d2_vd} Volts drop (erroneously calculating one-way conductor loop impedance without return path)"
        d3 = f"{d3_vd} Volts drop (erroneously applying an arbitrary 150 percent high-reactance scaling factor)"

        s = build_sample(q, correct, [d1, d2, d3], "nec_voltage_drop", rng)
        if s:
            samples.append(s)
    return samples


def generate_troubleshooting_diagnostics(rng, n=750):
    """Generate Multimeter & Field Troubleshooting Diagnostic Scenarios across industrial sites."""
    contexts = [
        "on a rooftop commercial HVAC chiller system",
        "in a manufacturing automated packaging conveyor cell",
        "at a municipal wastewater lift station pump panel",
        "inside a mission-critical hospital tier-3 data center",
        "on an industrial plastics extrusion molding line",
        "at a commercial cold-storage ammonia refrigeration unit",
        "in a chemical processing plant hazardous location",
    ]

    diagnostics = [
        (
            "An electrician measures 120V from hot to ground and 82V from hot to neutral at an active terminal block under load. At the distribution panel, both readings are exactly 120V. What is the fault?",
            "High-resistance joint or loose terminal on the neutral return branch conductor",
            "Direct short circuit between the hot ungrounded conductor and the metal enclosure",
            "An open equipment grounding safety conductor at the distribution panelboard",
            "Phase reversal on the primary winding of the utility step-down service transformer",
            "field_open_neutral",
        ),
        (
            "A high-impedance digital multimeter measures 68V AC on an isolated, de-energized control spare wire routed alongside energized 277V lighting feeders. A low-impedance (LoZ) meter reads 0.1V. What is the reading?",
            "Capacitively coupled phantom or ghost voltage induced from adjacent energized lines",
            "Direct dielectric insulation breakdown creating continuous leakage current to ground",
            "Severe harmonic circulating currents causing high-frequency line resonance on the spare",
            "A dangerous high-voltage line-to-frame fault within one of the luminaire ballasts",
            "field_ghost_voltage",
        ),
        (
            "A 460V 3-phase induction motor trips its thermal overload elements within 30 seconds of starting. Phase current measurements show Line 1 = 52A, Line 2 = 53A, Line 3 = 0A. What condition exists?",
            "Single-phasing condition caused by an open fuse or broken supply phase conductor on Line 3",
            "Severe three-phase reverse rotation due to swapped line phase connections on the starter",
            "Rotor bar fracture creating symmetric impedance imbalance across all three stator coils",
            "Thermal overload relay calibration drift causing premature tripping under normal load",
            "field_single_phasing",
        ),
        (
            "An AC motor magnetic contactor chatters loudly with intense 60Hz mechanical buzzing when the start button is engaged. What is the physical root cause?",
            "A fractured, loose, or missing copper shading coil on the laminated armature pole face",
            "Excessive line voltage applied across the contactor magnetic operating coil terminals",
            "Premature contact opening of the normally closed bimetallic overload relay elements",
            "Complete electrical short circuit across the main power current-carrying contact tips",
            "field_contactor_chatter",
        ),
        (
            "In a 3-wire start-stop motor control circuit, the starter contactor energizes only while the Start pushbutton is held pressed, and drops out immediately upon release. What component failed?",
            "The auxiliary normally open seal-in holding contact failed to close or has an open circuit",
            "The normally closed Stop pushbutton contact is mechanically seized in the open position",
            "The control circuit transformer primary fuse has blown due to excessive branch current",
            "The thermal overload heater contacts have permanently welded together in a closed state",
            "field_holding_circuit",
        ),
        (
            "An insulation resistance test using a 1,000V DC megohmmeter on a 480V three-phase stator winding reads 0.12 Megaohms after 1 minute. How should this result be interpreted?",
            "Severe insulation degradation and moisture ingress requiring reconditioning before energization",
            "Ideal baseline insulation resistance indicating pristine, dry, and healthy dielectric state",
            "Marginally acceptable reading satisfying standard IEEE 43 minimum resistance thresholds",
            "An erroneous reading caused by normal capacitive charging saturation of the stator core",
            "field_megohmmeter",
        ),
        (
            "A branch GFCI circuit breaker immediately trips when a 120V submersible pump is plugged in, even when the pump float switch is open and the motor is off. What fault is present?",
            "Neutral-to-ground fault on the pump supply cord or internal junction box wiring",
            "Severe branch circuit overload caused by excessive continuous steady-state current draw",
            "Harmonic distortion from adjacent variable frequency drive controllers corrupting sensing",
            "Line and load terminal conductors on the circuit breaker panel body are swapped backwards",
            "field_gfci_fault",
        ),
        (
            "In a 120/208V 3-phase 4-wire commercial panelboard feeding non-linear switching power supplies, a clamp meter reads 42A on Phase A, 44A on Phase B, 43A on Phase C, but 74A on the neutral. What causes this?",
            "Additive triplen harmonic currents (3rd, 9th, 15th) from single-phase non-linear power supplies",
            "Fundamental 60Hz load current imbalance shifting the neutral point away from electrical center",
            "Direct short circuit between one phase conductor and the grounded panelboard enclosure",
            "Circulating fault currents caused by missing grounding electrode conductor bonded at service",
            "field_harmonics",
        ),
    ]

    samples = []
    for _ in range(n):
        ctx = rng.choice(contexts)
        q_raw, corr, d1, d2, d3, cat = rng.choice(diagnostics)
        full_q = f"While troubleshooting electrical equipment {ctx}: {q_raw}"
        s = build_sample(full_q, corr, [d1, d2, d3], f"vocational_{cat}", rng)
        if s:
            samples.append(s)
    return samples


def main():
    print("================================================================")
    print(" GENERATING EXPANDED VOCATIONAL ELECTRICAL DATASET (TRACK D)")
    print("================================================================")

    rng = random.Random(42)

    generated = []
    generated.extend(generate_nec_continuous_loads(rng, n=750))
    generated.extend(generate_nec_ampacity_derating(rng, n=850))
    generated.extend(generate_nec_conduit_fill(rng, n=550))
    generated.extend(generate_nec_box_fill(rng, n=650))
    generated.extend(generate_nec_motor_protection(rng, n=700))
    generated.extend(generate_voltage_drop(rng, n=750))
    generated.extend(generate_troubleshooting_diagnostics(rng, n=750))

    print(f"Generated {len(generated)} grounded vocational electrical samples.")

    # Load existing pure electrical samples if available
    existing_samples = []
    if os.path.exists(TRACK_D_PURE_DIR):
        print(f"Loading existing pure samples from {TRACK_D_PURE_DIR}...")
        ds_pure = load_from_disk(TRACK_D_PURE_DIR)
        for split in ["train", "test"]:
            if split in ds_pure:
                for ex in ds_pure[split]:
                    existing_samples.append({
                        "question": ex["question"].strip(),
                        "choices": [str(c).strip() for c in ex["choices"]],
                        "text": ex["text"].strip(),
                        "label": int(ex["label"]),
                        "source": ex.get("source", "electrical_pure_base"),
                    })
        print(f"Loaded {len(existing_samples)} existing pure samples.")

    all_samples = existing_samples + generated
    print(f"\nTotal combined electrical samples before deduplication: {len(all_samples)}")

    # Deduplicate by normalized question
    seen = set()
    deduped = []
    for s in all_samples:
        norm_q = " ".join(s["question"].strip().lower().split())
        if norm_q not in seen:
            seen.add(norm_q)
            deduped.append(s)

    print(f"Unique electrical questions after deduplication: {len(deduped)}")

    # Deterministic split: 85% train, 15% test
    rng.shuffle(deduped)
    split_idx = int(len(deduped) * 0.85)
    train_data = deduped[:split_idx]
    test_data = deduped[split_idx:]

    print(f"\nFinal Split: {len(train_data)} train samples | {len(test_data)} test samples")

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
    })

    # Save to ./data/electrical_expanded
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_dict.save_to_disk(OUTPUT_DIR)
    print(f">>> Successfully saved expanded electrical dataset to {OUTPUT_DIR}")

    # Mirror to scratch/worktrees/track-d/data/electrical_expanded
    track_d_target = "./scratch/worktrees/track-d/data/electrical_expanded"
    os.makedirs(track_d_target, exist_ok=True)
    ds_dict.save_to_disk(track_d_target)
    print(f">>> Successfully mirrored expanded dataset to {track_d_target}")


if __name__ == "__main__":
    main()
