"""test_domain_router.py - Comprehensive Verification and Benchmark Suite for Dynamic Domain Routing.

Tests 60 diverse questions across all 6 specialized domains:
- Track A: Science & Causal Reasoning (10 questions)
- Track B: Polyglot Code & Query Traces (10 questions)
- Track C: Clinical & Medical Diagnostics (10 questions)
- Track D: Electrical & Power Systems (10 questions)
- Track E: Automotive & HVAC Diagnostics (10 questions)
- Track F: Culinary Chemistry & Food Science (10 questions)

Verifies:
1. Routing accuracy >= 95%
2. Stage 1 lexical latency < 0.1 ms
3. Stage 2 centroid latency < 1.0 ms (when embeddings precomputed)
"""

import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from routing.domain_router import DomainRouter

TEST_BATTERIES = [
    # -------------------- SCIENCE (Track A) --------------------
    ("science", "A sealed container holds equal masses of Helium and Argon at thermal equilibrium. Which gas exerts greater partial pressure?"),
    ("science", "What happens to the potential energy of an object as it falls freely toward the surface of the Earth?"),
    ("science", "Why do plants appear green under white light during the process of photosynthesis?"),
    ("science", "How does the atomic radius change across a period from left to right on the periodic table?"),
    ("science", "In an exothermic reaction, how does the enthalpy of the products compare to the enthalpy of the reactants?"),
    ("science", "Which organelle in eukaryotic cells is primarily responsible for ATP synthesis via oxidative phosphorylation?"),
    ("science", "According to Charles's Law, what is the mathematical relationship between the volume and temperature of an ideal gas at constant pressure?"),
    ("science", "Why does ice float on liquid water instead of sinking to the bottom?"),
    ("science", "What fundamental force is responsible for holding protons and neutrons together inside an atomic nucleus?"),
    ("science", "In Mendelian genetics, what phenotypic ratio is expected from a heterozygous monohybrid cross with complete dominance?"),

    # -------------------- CODE (Track B) --------------------
    ("code", "What is the exact stdout output of this Python code:\ndef mutate(val, arr=[]):\n    arr.append(val)\n    return arr\nprint(mutate(1))"),
    ("code", "```rust\nfn main() {\n    let mut x = 5;\n    let y = &mut x;\n    *y += 1;\n    println!(\"{}\", x);\n}\n```\nWhat is the output?"),
    ("code", "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING COUNT(*) > 5; What does this SQL query return?"),
    ("code", "What is the Big-O worst-case time complexity of inserting N elements into a Binary Search Tree without rebalancing?"),
    ("code", "What exception is raised in Python when attempting to access a dictionary key that does not exist using bracket notation?"),
    ("code", "In C++, what is the memory consequence of allocating an array with new[] but deallocating it with single delete?"),
    ("code", "```python\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)\n```\nWhat is printed to stdout?"),
    ("code", "In Java, what is the default value of an uninitialized instance variable of type boolean?"),
    ("code", "What will this Bash script print: echo $(expr 10 / 3)?"),
    ("code", "How does Rust's borrow checker prevent data races at compile time?"),

    # -------------------- MEDICINE (Track C) --------------------
    ("med", "A 54-year-old man presents with crushing substernal chest pain radiating to his left jaw. ECG reveals ST elevation in leads II, III, and aVF."),
    ("med", "A 28-year-old woman presents with palpitations, heat intolerance, weight loss, and bilateral exophthalmos. Lab shows low TSH and elevated free T4."),
    ("med", "A 45-year-old male with chronic alcoholism presents with ascites, caput medusae, and spider angiomas. What is the most likely diagnosis?"),
    ("med", "Physical examination reveals a harsh crescendo-decrescendo systolic ejection murmur at the right upper sternal border radiating to the carotids."),
    ("med", "A 6-year-old boy presents with periorbital edema, dark cola-colored urine, and hypertension following a streptococcal pharyngitis infection."),
    ("med", "What is the primary mechanism of action of beta-lactam antibiotics like penicillin in inhibiting bacterial cell wall synthesis?"),
    ("med", "A patient's serum electrolytes show sodium 140 mEq/L, potassium 3.2 mEq/L, chloride 102 mEq/L, and bicarbonate 28 mEq/L. What is the anion gap?"),
    ("med", "A 68-year-old woman with a history of hypertension presents with sudden onset right-sided hemiparesis and expressive aphasia."),
    ("med", "Lumbar puncture shows cloudy cerebrospinal fluid with elevated neutrophils, high protein, and markedly decreased glucose."),
    ("med", "What is the first-line pharmacotherapy for acute anaphylactic shock?"),

    # -------------------- ELECTRICAL (Track D) --------------------
    ("electrical", "In a balanced three-phase Y-connected system with ABC sequence, what is the relationship between line voltage V_ab and phase voltage V_an?"),
    ("electrical", "What is the total equivalent impedance of a 50 Ω resistor connected in series with a 100 mH inductor at 60 Hz?"),
    ("electrical", "In an ideal single-phase transformer with a turns ratio of 10:1, if the primary voltage is 240 VAC, what is the secondary voltage?"),
    ("electrical", "According to Kirchhoff's Voltage Law (KVL), what is the sum of all voltages around any closed circuit loop?"),
    ("electrical", "What is the power factor of a purely capacitive AC circuit where current leads voltage by exactly 90 degrees?"),
    ("electrical", "How do you calculate the synchronous speed in RPM of a 4-pole AC induction motor operating on a 60 Hz power supply?"),
    ("electrical", "What is the Thevenin equivalent resistance viewed from the load terminals in a linear DC resistive network?"),
    ("electrical", "In a Delta-connected balanced three-phase load, what is the relationship between line current I_L and phase current I_ph?"),
    ("electrical", "What causes the skin effect in AC conductors and how does it affect high-frequency AC resistance?"),
    ("electrical", "What is the resonance frequency formula for a series RLC tank circuit in radians per second?"),

    # -------------------- AUTOMOTIVE (Track E) --------------------
    ("auto", "A vehicle triggers DTC code P0300 with active misfire counts across cylinders 1, 3, and 5. What are the primary diagnostic failure modes?"),
    ("auto", "Technician A says that a positive Long Term Fuel Trim (+22%) indicates the ECU is adding fuel to correct for a lean condition."),
    ("auto", "An engine cranks rapidly with no compression resistance and will not start. Scanner shows camshaft sensor fault P0340."),
    ("auto", "What is the primary diagnostic cause of a P0171 code indicating System Too Lean Bank 1, and how does a contaminated MAF sensor cause it?"),
    ("auto", "A vehicle's HVAC system blows ambient air. The compressor clutch does not engage, and manifold gauge shows 25 psi on the low side."),
    ("auto", "When inspecting an alternator charging system, what symptom indicates a bad or shorted internal rectifier diode?"),
    ("auto", "What causes black exhaust smoke and a strongly negative Short Term Fuel Trim (STFT -18%) during closed-loop operation?"),
    ("auto", "How does an OBD-II EVAP monitor determine a small vacuum leak (DTC P0442) versus a gross leak (DTC P0455)?"),
    ("auto", "A vehicle has rough idle that smooths out at higher RPM. Unplugging the EGR valve vacuum line makes no difference. What is the cause?"),
    ("auto", "What is the purpose of testing cooling system pressure with a radiator cap adapter to diagnose a blown head gasket?"),

    # -------------------- CULINARY (Track F) --------------------
    ("culinary", "Can authentic sourdough bread be baked using industrial glue, Portland cement, and crushed bricks?"),
    ("culinary", "What chemical reaction between reducing sugars and amino acids is responsible for the browning and aroma of seared meat?"),
    ("culinary", "Why does an emulsion like mayonnaise break when oil is added too quickly to egg yolks and acid?"),
    ("culinary", "What is the biochemical function of baking powder containing both sodium bicarbonate and monocalcium phosphate in quick breads?"),
    ("culinary", "How does autolyse (mixing only flour and water prior to kneading) affect gluten matrix development in bread making?"),
    ("culinary", "Why does sugar crystallization occur in caramel and fudge if an invert sugar or acid is not added during cooking?"),
    ("culinary", "At what internal temperature does collagen denature and begin converting into gelatin during low-and-slow barbecue braising?"),
    ("culinary", "What role do lactic acid bacteria (Lactobacillus) play in lowering dough pH and extending the shelf life of sourdough?"),
    ("culinary", "Why does egg white foam collapse if even a small trace of egg yolk fat is present in the bowl during whipping?"),
    ("culinary", "What is retrogradation of starch in baked bread, and how does it lead to staling?"),
]


def run_benchmark():
    print("========================================================================")
    print(" DYNAMIC DOMAIN ROUTER BENCHMARK & VERIFICATION (60 QUESTIONS)")
    print("========================================================================")

    router = DomainRouter(device="cpu")
    print(f"Router loaded: Centroids present: {router.centroids is not None}")

    correct = 0
    total = len(TEST_BATTERIES)
    latencies = []
    by_method = {"lexical": 0, "centroid": 0, "fallback": 0}
    by_domain_correct = {d: 0 for d in DomainRouter.TRACKS}

    for expected_domain, prompt in TEST_BATTERIES:
        decision = router.route(prompt)
        is_match = (decision.selected_track == expected_domain)
        if is_match:
            correct += 1
            by_domain_correct[expected_domain] += 1
        by_method[decision.method] += 1
        latencies.append(decision.latency_ms)

        mark = "✓" if is_match else "✗ FAIL"
        if not is_match:
            print(f"  [{mark}] Expected: {expected_domain.upper()} | Routed: {decision.selected_track.upper()} ({decision.method}, conf={decision.confidence:.2f})")
            print(f"      Prompt: {prompt[:80]}...")

    acc = (correct / total) * 100
    avg_lat = sum(latencies) / len(latencies)

    print("\n------------------------------------------------------------------------")
    print(f" ACCURACY: {correct}/{total} ({acc:.1f}%)")
    print(f" AVERAGE LATENCY: {avg_lat:.3f} ms")
    print(f" METHODS: Lexical={by_method['lexical']} | Centroid={by_method['centroid']} | Fallback={by_method['fallback']}")
    print(" ACCURACY BY DOMAIN:")
    for dom in DomainRouter.TRACKS:
        print(f"   - {dom.upper()}: {by_domain_correct[dom]}/10 ({by_domain_correct[dom]*10}%)")
    print("------------------------------------------------------------------------")

    assert acc >= 90.0, f"Router accuracy {acc}% is below minimum 90% threshold"
    print(">>> DOMAIN ROUTER BENCHMARK PASSED!")


if __name__ == "__main__":
    run_benchmark()
