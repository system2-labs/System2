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
    ],
    "auto": [
        re.compile(r"\b[PBUC][0-3][0-9A-F]{3}\b"),  # OBD-II Diagnostic Trouble Codes (e.g. P0300, P0171)
        re.compile(r"\b(?:STFT|LTFT|fuel\s+trim|MAF\s+sensor|O2\s+sensor|MAP\s+sensor|TPS|crankshaft\s+sensor|camshaft\s+sensor)\b", re.IGNORECASE),
        re.compile(r"\b(?:misfire\s+cylinder|catalytic\s+converter|EVAP\s+purge|EGR\s+valve|alternator\s+diode|starter\s+solenoid)\b", re.IGNORECASE),
        re.compile(r"\b(?:OBD-?II|DTC\s+code|diagnostic\s+trouble\s+code|engine\s+hesitation|rough\s+idle|limp\s+mode)\b", re.IGNORECASE),
        re.compile(r"\b(?:ASE\s+Technician|Tech\s+A\s+says|Tech\s+B\s+says|HVAC\s+refrigerant|R-?134a|compressor\s+clutch)\b", re.IGNORECASE),
    ],
    "electrical": [
        re.compile(r"\b(?:three-phase|3-phase|line-to-line\s+voltage|phase\s+voltage|Y-connected|Wye-connected|Delta-connected)\b", re.IGNORECASE),
        re.compile(r"\b(?:phasor|impedance\s+triangle|apparent\s+power|reactive\s+power|power\s+factor|kVA|kVAR)\b", re.IGNORECASE),
        re.compile(r"\b(?:Kirchhoff|KVL|KCL|Thevenin|Norton|superposition\s+theorem|RLC\s+circuit)\b", re.IGNORECASE),
        re.compile(r"\b(?:transformer\s+turns\s+ratio|induction\s+motor\s+slip|synchronous\s+speed|skin\s+effect)\b", re.IGNORECASE),
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
    ],
    "culinary": [
        re.compile(r"\b(?:sourdough|leavening\s+agent|baking\s+powder|baking\s+soda|yeast\s+fermentation)\b", re.IGNORECASE),
        re.compile(r"\b(?:Maillard\s+reaction|caramelization|protein\s+denaturation|gluten\s+matrix|gluten\s+network)\b", re.IGNORECASE),
        re.compile(r"\b(?:emulsification|lecithin|mayonnaise\s+break|hollandaise|gelatinization|retrogradation)\b", re.IGNORECASE),
        re.compile(r"\b(?:proofing\s+dough|oven\s+spring|crumb\s+structure|baking\s+temperature|internal\s+temp\s+of\s+meat)\b", re.IGNORECASE),
        re.compile(r"\b(?:RecipeQA|culinary\s+chemistry|food\s+science|sugar\s+inversion|lactic\s+acid\s+bacteria)\b", re.IGNORECASE),
    ],
    "science": [
        re.compile(r"\b(?:molar\s+mass|Dalton'?s\s+law|ideal\s+gas|partial\s+pressure|thermal\s+equilibrium)\b", re.IGNORECASE),
        re.compile(r"\b(?:photosynthesis|cellular\s+respiration|gravitational\s+potential|kinetic\s+energy)\b", re.IGNORECASE),
        re.compile(r"\b(?:ARC-Challenge|causal\s+reasoning|hypothesis|experimental\s+control|independent\s+variable)\b", re.IGNORECASE),
        re.compile(r"\b(?:atomic\s+number|covalent\s+bond|ionic\s+bond|electronegativity|periodic\s+table)\b", re.IGNORECASE),
    ],
}


def scan_lexical_signatures(text: str) -> Optional[Tuple[str, float]]:
    """Scan prompt against high-precision technical signatures.

    Returns:
        (track_id, confidence) if a crisp signature is found with score >= 2, else None.
    """
    scores: Dict[str, int] = {k: 0 for k in PATTERNS}

    for domain, regex_list in PATTERNS.items():
        for regex in regex_list:
            matches = regex.findall(text)
            if matches:
                # Code blocks, OBD-II codes, or clinical vignette openings are decisive
                weight = 2 if (domain in ("code", "auto", "med") and len(matches) >= 1) else 1
                scores[domain] += len(matches) * weight

    best_domain = max(scores, key=lambda k: scores[k])
    best_score = scores[best_domain]

    # Require minimum activation score
    if best_score >= 2:
        # Confidence scaled by score margin
        other_scores = [v for k, v in scores.items() if k != best_domain]
        runner_up = max(other_scores) if other_scores else 0
        if best_score > runner_up:
            conf = min(0.99, 0.85 + 0.05 * (best_score - runner_up))
            return best_domain, conf

    return None
