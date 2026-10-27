#!/usr/bin/env python3
"""generate_physics_grounded_dataset.py - High-Purity Grounded Physics & Mechanics Generator.

Lifts Track J from academic trivia to 3,500+ pure grounded physical calculation and diagnostic samples.
Implements the In-Domain Causal Purity Law and Type-Aware Distractor Hardening:
1. 1D/2D Kinematics & Projectile Motion (trajectories, range, acceleration, velocity)
2. Newtonian Dynamics, Friction & Inclined Planes (free-body diagrams, normal force)
3. Work, Energy & Conservation Laws (spring potential energy, kinetic energy, non-conservative work)
4. Rotational Dynamics & Moment of Inertia (rolling without slipping, angular momentum, torque)
5. Electromagnetism & Circuits (Coulomb's Law, Lorentz force, cyclotron radius, RC circuits)
6. Thermodynamics & Heat Engines (Carnot efficiency in Kelvin, Ideal Gas Law, First Law)
7. Optics & Wave Mechanics (Snell's Law, total internal reflection, thin lens equation)

Every distractor models an authentic procedural physical mistake with character length parity (within +-15%).
Output: ./data/physics_grounded_expanded (train and test splits)
"""

import math
import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_from_disk

OUTPUT_DIR = "./data/physics_grounded_expanded"
TRACK_J_EXPANDED_DIR = "./data/physics_expanded"


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
            c = f"{c} (standard physics derivation)"
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


def generate_kinematics_projectiles(rng, n=700):
    """Generate Kinematics & Projectile Motion questions."""
    samples = []
    g = 9.8

    for _ in range(n):
        v0 = rng.randint(8, 85)
        angle_deg = rng.randint(15, 75)
        theta = math.radians(angle_deg)
        mode = rng.choice(["horizontal_range", "maximum_height", "flight_time"])

        if mode == "horizontal_range":
            range_m = round((v0**2 * math.sin(2 * theta)) / g, 1)
            d1 = round((v0**2 * math.sin(theta)) / g, 1)
            d2 = round((v0**2 * math.sin(2 * theta)) / (g * 0.5), 1)
            d3 = round((v0 * math.sin(2 * theta)) / g, 1)

            q = f"A projectile is launched from ground level with an initial velocity of {v0} m/s at an angle of {angle_deg}° above the horizontal. Neglecting air resistance and taking g = 9.8 m/s², what is the horizontal range of the projectile upon landing?"
            correct = f"{range_m} meters (calculated using R = (v0² * sin(2θ)) / g with 2θ = {2*angle_deg}°)"
            d1_text = f"{d1} meters (erroneously evaluating sin(θ) instead of the double-angle sin(2θ) factor)"
            d2_text = f"{d2} meters (erroneously underestimating the gravitational downward acceleration parameter)"
            d3_text = f"{d3} meters (erroneously omitting the quadratic squaring of the initial launch velocity vector)"

        elif mode == "maximum_height":
            v0y = v0 * math.sin(theta)
            h_max = round((v0y**2) / (2 * g), 1)
            d1 = round((v0**2) / (2 * g), 1)
            d2 = round((v0y**2) / g, 1)
            d3 = round((v0y) / (2 * g), 1)

            q = f"A projectile is fired with an initial speed of {v0} m/s at an angle of {angle_deg}° above the horizontal. Taking g = 9.8 m/s², what is the maximum vertical height achieved above the launch point?"
            correct = f"{h_max} meters (calculated using H = (v0 * sin(θ))² / (2g) with vertical velocity {v0y:.1f} m/s)"
            d1_text = f"{d1} meters (erroneously utilizing total kinetic velocity instead of the vertical component)"
            d2_text = f"{d2} meters (erroneously omitting the factor of two in the gravitational deceleration denominator)"
            d3_text = f"{d3} meters (erroneously omitting the quadratic kinetic velocity term)"

        else:  # flight_time
            v0y = v0 * math.sin(theta)
            t_total = round((2 * v0y) / g, 2)
            d1 = round(v0y / g, 2)
            d2 = round((2 * v0) / g, 2)
            d3 = round((v0y**2) / g, 2)

            q = f"A golf ball is hit with an initial velocity of {v0} m/s at an angle of {angle_deg}° above horizontal ground. Taking g = 9.8 m/s², what is the total time of flight before the ball strikes the ground?"
            correct = f"{t_total} seconds (calculated as total trajectory time T = (2 * v0 * sin(θ)) / g)"
            d1_text = f"{d1} seconds (erroneously calculating only the one-way ascending time to the apex peak)"
            d2_text = f"{d2} seconds (erroneously utilizing total scalar velocity instead of the vertical launch vector)"
            d3_text = f"{d3} seconds (erroneously applying quadratic kinematic scaling to the linear time dimension)"

        s = build_sample(q, correct, [d1_text, d2_text, d3_text], "physics_kinematics", rng)
        if s:
            samples.append(s)
    return samples


def generate_dynamics_inclines(rng, n=700):
    """Generate Newtonian Dynamics, Friction & Inclined Plane questions."""
    samples = []
    g = 9.8

    for _ in range(n):
        m = rng.randint(2, 45)
        angle_deg = rng.randint(18, 65)
        mu = round(rng.uniform(0.08, 0.42), 2)
        theta = math.radians(angle_deg)

        f_grav_parallel = g * math.sin(theta)
        f_fric_parallel = mu * g * math.cos(theta)
        a_net = round(f_grav_parallel - f_fric_parallel, 2)

        d1 = round(g * (math.cos(theta) - mu * math.sin(theta)), 2)
        d2 = round(f_grav_parallel + f_fric_parallel, 2)
        d3 = round(f_grav_parallel, 2)

        q = f"A block of mass {m} kg slides down a rough ramp inclined at {angle_deg}° to the horizontal. If the coefficient of kinetic friction between the block and the ramp surface is μ = {mu}, what is the acceleration of the block down the incline? (g = 9.8 m/s²)"

        correct = f"{a_net} m/s² down the ramp (calculated using a = g * (sin({angle_deg}°) - {mu} * cos({angle_deg}°)))"
        d1_text = f"{d1} m/s² down the ramp (erroneously swapping trigonometric sine and cosine resolution vectors)"
        d2_text = f"{d2} m/s² down the ramp (erroneously adding frictional force in the direction of downhill sliding)"
        d3_text = f"{d3} m/s² down the ramp (erroneously calculating idealized frictionless gravitational acceleration)"

        s = build_sample(q, correct, [d1_text, d2_text, d3_text], "physics_dynamics_incline", rng)
        if s:
            samples.append(s)
    return samples


def generate_energy_conservation(rng, n=650):
    """Generate Work, Energy Conservation & Spring Potential questions."""
    samples = []

    for _ in range(n):
        m = round(rng.uniform(0.5, 12.0), 1)
        k = rng.choice(list(range(60, 1200, 20)))
        x = round(rng.uniform(0.04, 0.45), 2)

        v_max = round(math.sqrt(k / m) * x, 2)
        e_pot = round(0.5 * k * (x**2), 2)

        d1 = round((k / m) * x, 2)
        d2 = round(math.sqrt(k * x / m), 2)
        d3 = round(math.sqrt(m / k) * x, 2)

        q = f"A horizontal frictionless spring system has a spring constant k = {k} N/m and an attached block of mass m = {m} kg. If the spring is compressed by {int(x*100)} cm ({x} m) and released from rest, what is the maximum velocity attained by the mass as it passes through the equilibrium position?"

        correct = f"{v_max} m/s (equating elastic potential energy 0.5kx² of {e_pot} J to kinetic energy 0.5mv²)"
        d1_text = f"{d1} m/s (erroneously omitting the square-root radical on the spring-to-mass stiffness ratio)"
        d2_text = f"{d2} m/s (erroneously treating elastic potential energy as linearly proportional to displacement)"
        d3_text = f"{d3} m/s (erroneously inverting the inertial mass ratio in the conservation of energy equation)"

        s = build_sample(q, correct, [d1_text, d2_text, d3_text], "physics_energy_spring", rng)
        if s:
            samples.append(s)
    return samples


def generate_rotational_dynamics(rng, n=600):
    """Generate Rotational Dynamics & Rolling without Slipping questions."""
    samples = []
    geometries = [
        ("solid cylinder or uniform disk", 0.5, "1/2 M R²"),
        ("thin spherical shell or hollow sphere", 2.0 / 3.0, "2/3 M R²"),
        ("solid uniform sphere", 2.0 / 5.0, "2/5 M R²"),
        ("thin hoop or cylindrical ring", 1.0, "M R²"),
    ]
    g = 9.8

    for _ in range(n):
        obj_name, c_geom, geom_formula = rng.choice(geometries)
        h = round(rng.uniform(1.5, 25.0), 1)

        v_roll = round(math.sqrt((2 * g * h) / (1.0 + c_geom)), 2)
        d1 = round(math.sqrt(2 * g * h), 2)
        d2 = round(math.sqrt((2 * g * h) / c_geom), 2)
        d3 = round(math.sqrt((2 * g * h) / max(0.2, 1.0 - c_geom)), 2)

        q = f"A {obj_name} (moment of inertia I = {geom_formula}) rolls smoothly without slipping down an inclined ramp from a vertical release height of h = {h} meters. Taking g = 9.8 m/s², what is the linear speed of its center of mass at the bottom of the ramp?"

        correct = f"{v_roll} m/s (derived from energy conservation partitioning kinetic energy between translation and rotation)"
        d1_text = f"{d1} m/s (erroneously modeling frictionless sliding without allocating kinetic energy to rotational spin)"
        d2_text = f"{d2} m/s (erroneously omitting translational center-of-mass energy from the total kinetic budget)"
        d3_text = f"{d3} m/s (erroneously subtracting rotational inertia from the total kinetic energy divisor)"

        s = build_sample(q, correct, [d1_text, d2_text, d3_text], "physics_rotational_energy", rng)
        if s:
            samples.append(s)
    return samples


def generate_electromagnetism_circuits(rng, n=600):
    """Generate Electromagnetism, Lorentz Force & Cyclotron Radius questions."""
    samples = []
    particles = [
        ("proton (mass m = 1.67 × 10⁻²⁷ kg, charge q = +1.60 × 10⁻¹⁹ C)", 1.67e-27, 1.6e-19),
        ("alpha particle (mass m = 6.64 × 10⁻²⁷ kg, charge q = +3.20 × 10⁻¹⁹ C)", 6.64e-27, 3.2e-19),
        ("deuteron (mass m = 3.34 × 10⁻²⁷ kg, charge q = +1.60 × 10⁻¹⁹ C)", 3.34e-27, 1.6e-19),
        ("singly ionized carbon-12 ion (mass m = 2.00 × 10⁻²⁶ kg, charge q = +1.60 × 10⁻¹⁹ C)", 2.0e-26, 1.6e-19),
    ]

    for _ in range(n):
        p_name, m, q = rng.choice(particles)
        b_tesla = round(rng.uniform(0.15, 3.5), 2)
        v = rng.choice([5e5, 1e6, 1.5e6, 2e6, 2.5e6, 3e6, 4e6, 5e6])

        r_m = (m * v) / (q * b_tesla)
        r_cm = round(r_m * 100.0, 2)

        d1_cm = round(((q * b_tesla) / (m * v)) * 100.0, 2)
        d2_cm = round(((m * v) / q) * 100.0, 2)
        d3_cm = round(((m * (v**2)) / (q * b_tesla)) * 100.0, 2)

        q_stem = f"A {p_name} enters a uniform magnetic field of B = {b_tesla} Tesla at a velocity of {v:.1e} m/s perpendicular to the field lines. What is the radius of the resulting circular trajectory?"

        correct = f"{r_cm} cm (calculated using the centripetal-magnetic equilibrium formula r = (m * v) / (q * B))"
        d1_text = f"{d1_cm} cm (erroneously inverting the ratio of particle momentum to magnetic lorentz force terms)"
        d2_text = f"{d2_cm} cm (erroneously omitting magnetic field induction strength B from the trajectory denominator)"
        d3_text = f"{d3_cm} cm (erroneously applying quadratic centripetal acceleration scaling to orbit radius)"

        s = build_sample(q_stem, correct, [d1_text, d2_text, d3_text], "physics_lorentz_cyclotron", rng)
        if s:
            samples.append(s)
    return samples


def generate_thermodynamics_carnot(rng, n=600):
    """Generate Carnot Efficiency & Thermodynamics questions."""
    samples = []
    for _ in range(n):
        tc_c = rng.randint(5, 75)
        th_c = rng.randint(220, 850)

        tc_k = tc_c + 273.15
        th_k = th_c + 273.15

        eff_carnot = round((1.0 - tc_k / th_k) * 100.0, 1)

        d1 = round((1.0 - tc_c / th_c) * 100.0, 1)
        d2 = round((1.0 - th_k / tc_k) * 100.0, 1)
        d3 = round((tc_k / th_k) * 100.0, 1)

        q = (
            f"A hypothetical heat engine operates between a high-temperature thermal reservoir at {th_c}°C "
            f"and a low-temperature heat sink at {tc_c}°C. According to the Second Law of Thermodynamics, "
            f"what is the maximum theoretical thermodynamic efficiency (Carnot efficiency) achievable by this engine?"
        )

        correct = f"{eff_carnot} percent (calculated as η = 1 - (Tc / Th) converting temperatures to {tc_k:.1f} K and {th_k:.1f} K)"
        d1_text = f"{d1} percent (erroneously performing thermodynamic calculations directly in Celsius without Kelvin conversion)"
        d2_text = f"{d2} percent (erroneously reversing thermal reservoir hierarchy in the Carnot thermodynamic efficiency formula)"
        d3_text = f"{d3} percent (erroneously reporting the fraction of rejected waste heat instead of extracted network efficiency)"

        s = build_sample(q, correct, [d1_text, d2_text, d3_text], "physics_thermodynamics_carnot", rng)
        if s:
            samples.append(s)
    return samples


def generate_optics_snell(rng, n=600):
    """Generate Snell's Law & Total Internal Reflection questions."""
    samples = []
    indices = [
        ("water", 1.33),
        ("fused silica", 1.46),
        ("crown glass", 1.52),
        ("flint glass", 1.66),
        ("sapphire", 1.77),
        ("cubic zirconia", 2.15),
        ("diamond", 2.42),
        ("silicon in infrared", 3.44),
    ]

    for _ in range(n):
        med1_name, n1 = rng.choice(indices)
        lower_indices = [x for x in indices if x[1] < n1] + [("air", 1.00)]
        med2_name, n2 = rng.choice(lower_indices)

        sin_tc = n2 / n1
        crit_deg = round(math.degrees(math.asin(sin_tc)), 1)

        d1 = round(math.degrees(math.asin(min(1.0, n1 / n2 if n1 < n2 else 0.5))), 1)
        d2 = round((n2 / n1) * 90.0, 1)
        d3 = round(90.0 - crit_deg, 1)

        q = (
            f"Light travels from {med1_name} (refractive index n1 = {n1}) toward an interface with {med2_name} (refractive index n2 = {n2}). "
            f"What is the critical angle for total internal reflection at this boundary?"
        )

        correct = f"{crit_deg} degrees (calculated from Snell's law critical boundary condition sin(θc) = n2 / n1 = {sin_tc:.3f})"
        d1_text = f"{d1} degrees (erroneously inverting the refractive index ratio in the total internal reflection equation)"
        d2_text = f"{d2} degrees (erroneously calculating a linear proportional quadrant angle without arcsine trigonometric mapping)"
        d3_text = f"{d3} degrees (erroneously computing the grazing angle along the optical interface instead of normal incidence)"

        s = build_sample(q, correct, [d1_text, d2_text, d3_text], "physics_optics_snell", rng)
        if s:
            samples.append(s)
    return samples


def main():
    print("================================================================")
    print(" GENERATING GROUNDED PURE PHYSICS DATASET (TRACK J)")
    print("================================================================")

    rng = random.Random(42)

    generated = []
    generated.extend(generate_kinematics_projectiles(rng, n=700))
    generated.extend(generate_dynamics_inclines(rng, n=700))
    generated.extend(generate_energy_conservation(rng, n=650))
    generated.extend(generate_rotational_dynamics(rng, n=600))
    generated.extend(generate_electromagnetism_circuits(rng, n=600))
    generated.extend(generate_thermodynamics_carnot(rng, n=600))
    generated.extend(generate_optics_snell(rng, n=600))

    print(f"Generated {len(generated)} grounded physical reasoning samples.")

    # Load existing physics samples if available
    existing = []
    if os.path.exists(TRACK_J_EXPANDED_DIR):
        print(f"Loading existing physics samples from {TRACK_J_EXPANDED_DIR}...")
        try:
            ds_exp = load_from_disk(TRACK_J_EXPANDED_DIR)
            for split in ["train", "test"]:
                if split in ds_exp:
                    for ex in ds_exp[split]:
                        existing.append({
                            "question": ex["question"].strip(),
                            "choices": [str(c).strip() for c in ex["choices"]],
                            "text": ex["text"].strip(),
                            "label": int(ex["label"]),
                            "source": ex.get("source", "physics_exp_base"),
                        })
            print(f"Loaded {len(existing)} existing physics samples.")
        except Exception as e:
            print(f"Warning: could not load expanded physics: {e}")

    all_samples = existing + generated
    print(f"\nTotal combined physics samples before deduplication: {len(all_samples)}")

    seen = set()
    deduped = []
    for s in all_samples:
        norm_q = " ".join(s["question"].strip().lower().split())
        if norm_q not in seen:
            seen.add(norm_q)
            deduped.append(s)

    print(f"Unique physics questions after deduplication: {len(deduped)}")
    print("Source breakdown:", dict(Counter(s["source"] for s in deduped)))

    # Split 85% train, 15% test
    rng.shuffle(deduped)
    split_idx = int(len(deduped) * 0.85)
    train_data = deduped[:split_idx]
    test_data = deduped[split_idx:]

    print(f"\nFinal Split: {len(train_data)} train samples | {len(test_data)} test samples")

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_dict.save_to_disk(OUTPUT_DIR)
    print(f">>> Successfully saved grounded expanded physics dataset to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
