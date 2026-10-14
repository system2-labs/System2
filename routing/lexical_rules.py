"""lexical_rules.py - Stage 1 Ultra-Fast Lexical Gate for System2 Dynamic Routing.

Executes deterministic pattern and keyword scanning in < 0.05 ms before touching neural models.
"""

import re
from typing import Dict, List, Optional, Tuple

# Domain signatures and compiled regexes
PATTERNS = {
    "code": [
        re.compile(r"```(?:python|rust|cpp|c|csharp|java|sql|bash|sh|go|typescript|javascript)\b", re.IGNORECASE),
        re.compile(r"\b(?:def\s+[a-zA-Z_]\w*\(|fn\s+[a-zA-Z_]\w*\(|function\s+[a-zA-Z_]\w*\(|public\s+static\s+void)\b"),
        re.compile(r"\b(?:SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+.+\s+SET|DELETE\s+FROM)\b", re.IGNORECASE),
        re.compile(r"\b(?:stdout\s+output|exact\s+output|runtime\s+complexity|Big-O|stack\s+overflow|segmentation\s+fault)\b", re.IGNORECASE),
        re.compile(r"\b(?:arr\.append|println!|console\.log|printf\(|std::cout|nullptr|malloc\(|free\()\b"),
        re.compile(r"(?:\b(?:borrow\s+checker|data\s+races|instance\s+variable|Binary\s+Search\s+Tree|BST|dictionary\s+key)\b|\bC\+\+|\bnew\[\]|\bdelete\[\])"),
        re.compile(r"\bIn\s+(?:Python|Rust|Java|Bash|SQL|Golang|C\+\+)", re.IGNORECASE),
    ],
    "auto": [
        re.compile(r"\b[PBUC][0-3][0-9A-F]{3}\b"),  # OBD-II Diagnostic Trouble Codes (e.g. P0300, P0171)
        re.compile(r"\b(?:STFT|LTFT|fuel\s+trim|MAF\s+sensor|O2\s+sensor|MAP\s+sensor|TPS|crankshaft\s+sensor|camshaft\s+sensor)\b", re.IGNORECASE),
        re.compile(r"\b(?:misfire\s+cylinder|catalytic\s+converter|EVAP\s+purge|EGR\s+valve|alternator\s+diode|starter\s+solenoid)\b", re.IGNORECASE),
        re.compile(r"\b(?:OBD-?II|DTC\s+code|diagnostic\s+trouble\s+code|engine\s+hesitation|rough\s+idle|limp\s+mode)\b", re.IGNORECASE),
        re.compile(r"\b(?:ASE\s+Technician|Tech\s+A\s+says|Tech\s+B\s+says|HVAC\s+refrigerant|R-?134a|compressor\s+clutch)\b", re.IGNORECASE),
        re.compile(r"\b(?:radiator\s+cap|blown\s+head\s+gasket|exhaust\s+smoke|cranks\s+rapidly|no\s+compression)\b", re.IGNORECASE),
    ],
    "electrical": [
        re.compile(r"\b(?:three-phase|3-phase|line-to-line\s+voltage|phase\s+voltage|Y-connected|Wye-connected|Delta-connected)\b", re.IGNORECASE),
        re.compile(r"\b(?:phasor|impedance\s+triangle|apparent\s+power|reactive\s+power|power\s+factor|kVA|kVAR)\b", re.IGNORECASE),
        re.compile(r"\b(?:Kirchhoff|KVL|KCL|Thevenin|Norton|superposition\s+theorem|RLC\s+circuit|RLC\s+tank)\b", re.IGNORECASE),
        re.compile(r"\b(?:transformer\s+turns\s+ratio|induction\s+motor|synchronous\s+speed|skin\s+effect|4-pole|stator|rotor)\b", re.IGNORECASE),
        re.compile(r"\b(?:V_ab|V_an|V_bc|V_cn|I_a|I_b|I_c)\b"),
        re.compile(r"(?:[0-9]+(?:\.[0-9]+)?\s*(?:k?Ω|uF|μF|mF|mH|uH|μH|kHz|MHz|VAC|VDC))"),
    ],
    "med": [
        re.compile(r"\b(?:[0-9]+-year-old\s+(?:man|woman|boy|girl|male|female)\s+presents\s+with)\b", re.IGNORECASE),
        re.compile(r"\b(?:vital\s+signs\s+show|physical\s+examination\s+reveals|auscultation\s+reveals)\b", re.IGNORECASE),
        re.compile(r"\b(?:systolic\s+ejection\s+murmur|holosystolic\s+murmur|cranial\s+nerve|cerebrospinal\s+fluid)\b", re.IGNORECASE),
        re.compile(r"\b(?:differential\s+diagnosis|most\s+likely\s+diagnosis|pharmacotherapy|mechanism\s+of\s+action)\b", re.IGNORECASE),
        re.compile(r"\b(?:USMLE|MedQA|board\s+examination|histopathology|biopsy\s+shows|Gram-negative|Gram-positive)\b", re.IGNORECASE),
        re.compile(r"\b(?:mg/dL|mEq/L|mmHg|g/dL|U/L|mmol/L)\b"),
        re.compile(r"\b(?:anaphylactic\s+shock|lumbar\s+puncture|hemiparesis|aphasia|periorbital\s+edema)\b", re.IGNORECASE),
    ],
    "culinary": [
        re.compile(r"\b(?:sourdough|leavening\s+agent|baking\s+powder|baking\s+soda|yeast\s+fermentation)\b", re.IGNORECASE),
        re.compile(r"\b(?:Maillard\s+reaction|caramelization|protein\s+denaturation|gluten\s+matrix|gluten\s+network)\b", re.IGNORECASE),
        re.compile(r"\b(?:emulsification|lecithin|mayonnaise\s+break|mayonnaise|hollandaise|gelatinization|retrogradation)\b", re.IGNORECASE),
        re.compile(r"\b(?:proofing\s+dough|oven\s+spring|crumb\s+structure|baking\s+temperature|internal\s+temp\s+of\s+meat)\b", re.IGNORECASE),
        re.compile(r"\b(?:RecipeQA|culinary\s+chemistry|food\s+science|sugar\s+inversion|lactic\s+acid\s+bacteria)\b", re.IGNORECASE),
        re.compile(r"\b(?:autolyse|sugar\s+crystallization|collagen\s+denature|converting\s+into\s+gelatin|egg\s+white\s+foam)\b", re.IGNORECASE),
        re.compile(r"\b(?:baked\s+bread|seared\s+meat|kneading|staling)\b", re.IGNORECASE),
    ],
    "science": [
        re.compile(r"\b(?:molar\s+mass|Dalton'?s\s+law|ideal\s+gas|partial\s+pressure|thermal\s+equilibrium)\b", re.IGNORECASE),
        re.compile(r"\b(?:photosynthesis|cellular\s+respiration|gravitational\s+potential|kinetic\s+energy)\b", re.IGNORECASE),
        re.compile(r"\b(?:ARC-Challenge|causal\s+reasoning|hypothesis|experimental\s+control|independent\s+variable)\b", re.IGNORECASE),
        re.compile(r"\b(?:atomic\s+number|atomic\s+radius|covalent\s+bond|ionic\s+bond|electronegativity|periodic\s+table)\b", re.IGNORECASE),
        re.compile(r"\b(?:Charles'?s\s+Law|exothermic\s+reaction|enthalpy|oxidative\s+phosphorylation|Mendelian\s+genetics)\b", re.IGNORECASE),
    ],
    "math": [
        re.compile(r"\b(?:GSM8K|word\s+problem|arithmetic\s+sequence|algebraic\s+expression|solve\s+for\s+x)\b", re.IGNORECASE),
        re.compile(r"\b(?:how\s+many\s+altogether|how\s+much\s+money|how\s+many\s+miles|how\s+many\s+hours|how\s+many\s+days|how\s+many\s+clips|how\s+many\s+apples)\b", re.IGNORECASE),
        re.compile(r"\b(?:quadratic\s+equation|polynomial|Pythagorean\s+theorem|prime\s+factorization|factorial)\b", re.IGNORECASE),
        re.compile(r"\b(?:percentage\s+increase|cost\s+per\s+item|discount\s+of|calculate\s+the\s+total)\b", re.IGNORECASE),
    ],
    "legal": [
        re.compile(r"\b(?:torts?|statute|breach\s+of\s+contract|jurisdiction|mens\s+rea|actus\s+reus)\b", re.IGNORECASE),
        re.compile(r"\b(?:plaintiff|defendant|prosecution|appellant|appellee|habeas\s+corpus)\b", re.IGNORECASE),
        re.compile(r"\b(?:Fifth\s+Amendment|Fourteenth\s+Amendment|due\s+process|strict\s+liability|negligence)\b", re.IGNORECASE),
        re.compile(r"\b(?:hearsay\s+exception|parol\s+evidence|sovereign\s+immunity|summary\s+judgment)\b", re.IGNORECASE),
        re.compile(r"\b(?:Bar\s+Exam|Multistate\s+Bar|MBE|jurisprudence|fiduciary\s+duty)\b", re.IGNORECASE),
    ],
    "finance": [
        re.compile(r"\b(?:depreciation|EBITDA|balance\s+sheet|income\s+statement|cash\s+flow)\b", re.IGNORECASE),
        re.compile(r"\b(?:macroeconomic|microeconomic|econometrics|monetary\s+policy|inflation\s+rate)\b", re.IGNORECASE),
        re.compile(r"\b(?:amortization|capital\s+structure|weighted\s+average\s+cost|WACC|liquidity\s+ratio)\b", re.IGNORECASE),
        re.compile(r"\b(?:gross\s+margin|accounts\s+receivable|accounts\s+payable|equity\s+multiplier)\b", re.IGNORECASE),
        re.compile(r"\b(?:fiscal\s+stimulus|Keynesian|price\s+elasticity|comparative\s+advantage|GDP\s+deflator)\b", re.IGNORECASE),
    ],
}


def scan_lexical_signatures(text: str) -> Optional[Tuple[str, float]]:
    """Scan prompt against high-precision technical signatures.

    Returns:
        (track_id, confidence) if a signature is found with score >= 1 and distinct margin.
    """
    scores: Dict[str, int] = {k: 0 for k in PATTERNS}

    for domain, regex_list in PATTERNS.items():
        for regex in regex_list:
            matches = regex.findall(text)
            if matches:
                # Decisive domain matches carry score 2
                weight = 2
                scores[domain] += len(matches) * weight

    best_domain = max(scores, key=lambda k: scores[k])
    best_score = scores[best_domain]

    if best_score >= 1:
        other_scores = [v for k, v in scores.items() if k != best_domain]
        runner_up = max(other_scores) if other_scores else 0
        if best_score > runner_up:
            conf = min(0.99, 0.88 + 0.04 * (best_score - runner_up))
            return best_domain, conf

    return None
