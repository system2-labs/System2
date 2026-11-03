#!/usr/bin/env python3
"""generate_finance_grounded_dataset.py - High-Purity Grounded Financial Reasoning Generator.

Lifts Track I from 255 samples to 3,500+ pure grounded financial samples.
Implements the In-Domain Causal Purity Law and Type-Aware Distractor Hardening:
1. WACC, CAPM & Cost of Capital (debt tax shields, beta unlevering/relevering)
2. DuPont ROE Decomposition (3-stage and 5-stage ratios, asset turnover, leverage)
3. Fixed Income: Bond Pricing, YTM, Macaulay & Modified Duration, Convexity
4. Capital Budgeting: NPV, IRR, Payback Period, and Profitability Index
5. Corporate Valuation: FCF, Gordon Growth Model, Enterprise Value vs Equity Value
6. Options & Derivatives: Put-Call Parity, Option Spreads, and Breakeven Payoffs
7. Working Capital & Liquidity: Cash Conversion Cycle, Quick Ratio, DSO/DIO/DPO

Every distractor models an authentic procedural mistake with character length parity (within +-15%).
Output: ./data/finance_expanded (train and test splits)
"""

import math
import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_from_disk

OUTPUT_DIR = "./data/finance_expanded"
TRACK_I_BASE_DIR = "./scratch/worktrees/track-i/data/finance"


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

    # Length parity padding if significantly shorter
    max_len = max(len(c) for c in choices)
    norm_choices = []
    for c in choices:
        if len(c) < max_len * 0.82:
            c = f"{c} (standard corporate finance calculation)"
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


def generate_wacc_capm(rng, n=600):
    """Generate WACC and CAPM calculation questions."""
    samples = []
    rf_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    mkt_premiums = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    betas = [0.75, 0.90, 1.05, 1.20, 1.35, 1.50]
    tax_rates = [21.0, 25.0, 28.0, 30.0]
    cost_debts = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    weights_equity = [0.40, 0.50, 0.60, 0.70, 0.80]

    for _ in range(n):
        rf = rng.choice(rf_values)
        erp = rng.choice(mkt_premiums)
        b = rng.choice(betas)
        tax = rng.choice(tax_rates)
        rd = rng.choice(cost_debts)
        we = rng.choice(weights_equity)
        wd = round(1.0 - we, 2)

        # CAPM Cost of Equity
        cost_equity = round(rf + b * erp, 2)
        # WACC = We * Re + Wd * Rd * (1 - t)
        after_tax_rd = round(rd * (1.0 - tax / 100.0), 3)
        wacc = round(we * cost_equity + wd * after_tax_rd, 2)

        # Distractor 1: forgot debt tax shield (used pretax Rd)
        d1_wacc = round(we * cost_equity + wd * rd, 2)
        # Distractor 2: swapped debt and equity weights
        d2_wacc = round(wd * cost_equity + we * after_tax_rd, 2)
        # Distractor 3: miscalculated CAPM (treated ERP as expected return, subtracting rf twice)
        err_re = round(rf + b * (erp - rf), 2)
        d3_wacc = round(we * err_re + wd * after_tax_rd, 2)

        q = (
            f"A corporation has a capital structure with {int(we*100)}% equity and {int(wd*100)}% debt. "
            f"The risk-free rate is {rf}%, the market risk premium is {erp}%, and the firm's equity beta is {b}. "
            f"If the pre-tax cost of debt is {rd}% and the corporate tax rate is {int(tax)}%, what is the firm's Weighted Average Cost of Capital (WACC)?"
        )

        correct = f"{wacc:.2f} percent (calculated using the after-tax cost of debt and CAPM equity cost of {cost_equity:.2f}%)"
        d1 = f"{d1_wacc:.2f} percent (erroneously omitting the interest expense tax shield on the cost of debt)"
        d2 = f"{d2_wacc:.2f} percent (erroneously reversing the proportional weights of equity and debt financing)"
        d3 = f"{d3_wacc:.2f} percent (erroneously subtracting the risk-free rate from an already net market risk premium)"

        s = build_sample(q, correct, [d1, d2, d3], "finance_wacc_capm", rng)
        if s:
            samples.append(s)
    return samples


def generate_dupont_roe(rng, n=550):
    """Generate 3-stage and 5-stage DuPont ROE decomposition questions."""
    samples = []
    for _ in range(n):
        net_income = rng.choice([25, 40, 60, 80, 100, 120, 150, 200])  # in millions
        sales = net_income * rng.choice([5, 8, 10, 12, 15])
        assets = sales * rng.choice([0.6, 0.8, 1.0, 1.2, 1.5])
        equity = assets * rng.choice([0.3, 0.4, 0.5, 0.6])

        npm = (net_income / sales) * 100.0
        tat = sales / assets
        em = assets / equity
        roe = round((npm / 100.0) * tat * em * 100.0, 2)

        # Distractor 1: used ROA instead of ROE (omitted equity multiplier)
        d1_roe = round((npm / 100.0) * tat * 100.0, 2)
        # Distractor 2: used Debt/Equity ratio instead of Assets/Equity for multiplier
        debt = assets - equity
        d2_roe = round((npm / 100.0) * tat * (debt / equity) * 100.0, 2)
        # Distractor 3: inverted asset turnover (Assets / Sales)
        d3_roe = round((npm / 100.0) * (assets / sales) * em * 100.0, 2)

        q = (
            f"A firm reports Net Income of ${net_income} million, Total Revenue of ${sales:.1f} million, "
            f"Total Assets of ${assets:.1f} million, and Shareholders' Equity of ${equity:.1f} million. "
            f"Using the three-component DuPont formula, what is the firm's Return on Equity (ROE)?"
        )

        correct = f"{roe:.2f} percent (product of {npm:.2f}% net profit margin, {tat:.2f} asset turnover, and {em:.2f} equity multiplier)"
        d1 = f"{d1_roe:.2f} percent (erroneously computing Return on Assets by omitting the equity multiplier leverage factor)"
        d2 = f"{d2_roe:.2f} percent (erroneously utilizing the debt-to-equity ratio instead of total assets-to-equity leverage)"
        d3 = f"{d3_roe:.2f} percent (erroneously inverting capital efficiency by dividing total assets by annual sales)"

        s = build_sample(q, correct, [d1, d2, d3], "finance_dupont_roe", rng)
        if s:
            samples.append(s)
    return samples


def generate_bond_pricing_duration(rng, n=600):
    """Generate Bond Pricing, Yield to Maturity, and Modified Duration questions."""
    samples = []
    coupons = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ytms = [3.5, 4.5, 5.5, 6.5, 7.5]
    maturities = [5, 7, 10, 15, 20]

    for _ in range(n):
        c_rate = rng.choice(coupons)
        y = rng.choice(ytms)
        n_years = rng.choice(maturities)
        mode = rng.choice(["price_direction", "modified_duration_shock", "annual_coupon"])

        if mode == "price_direction":
            price_status = "discount" if y > c_rate else "premium" if y < c_rate else "par"
            q = (
                f"A corporate bond has a coupon rate of {c_rate}% and a yield to maturity (YTM) of {y}%. "
                f"All else equal, is this bond currently trading at a premium, discount, or par value, and how does its current price relate to its $1,000 face value?"
            )
            if price_status == "discount":
                correct = f"Trading at a discount below $1,000 face value (required market yield of {y}% exceeds the {c_rate}% stated coupon)"
                d1 = f"Trading at a premium above $1,000 face value (investors pay more to lock in the stated annual coupon cash flow)"
                d2 = f"Trading exactly at par value of $1,000 (fixed income arbitrage keeps market price tied to redemption value)"
                d3 = f"Trading at a variable floating premium based strictly on the issuer's subordinated credit default spread"
            else:
                correct = f"Trading at a premium above $1,000 face value (stated coupon of {c_rate}% exceeds the market required yield of {y}%)"
                d1 = f"Trading at a discount below $1,000 face value (investors require higher initial yields to compensate for reinvestment)"
                d2 = f"Trading exactly at par value of $1,000 (fixed income arbitrage keeps market price tied to redemption value)"
                d3 = f"Trading at a variable zero-coupon discount governed exclusively by benchmark treasury curve shift"

        elif mode == "modified_duration_shock":
            mac_dur = round(n_years * 0.75, 2)
            mod_dur = round(mac_dur / (1.0 + y / 100.0), 2)
            bps_shift = rng.choice([50, 75, 100, 150])
            delta_y = bps_shift / 10000.0
            pct_change = round(-mod_dur * (delta_y * 100.0), 2)

            q = (
                f"A portfolio manager holds a 10-year corporate bond with a Modified Duration of {mod_dur} years. "
                f"If benchmark interest rates increase by {bps_shift} basis points (+{bps_shift/100.0}%), what is the approximate percentage change in the bond's price?"
            )
            correct = f"{pct_change:.2f} percent (price decreases by approximately {abs(pct_change):.2f}% due to the inverse duration relation)"
            d1 = f"{abs(pct_change):.2f} percent (price increases by approximately {abs(pct_change):.2f}% reflecting higher yield valuation)"
            d2 = f"{round(pct_change * 1.5, 2):.2f} percent (erroneously scaling price sensitivity by the nominal maturity instead of duration)"
            d3 = f"{round(pct_change * 0.5, 2):.2f} percent (erroneously dividing price volatility by semi-annual compounding factor squared)"

        else:  # annual coupon cash flow
            face = 1000
            annual_c = int(face * (c_rate / 100.0))
            semi_c = round(annual_c / 2.0, 2)
            q = (
                f"An investor purchases a $1,000 par value bond paying an annual coupon rate of {c_rate}% with semi-annual coupon distribution. "
                f"What cash flow does the investor receive every six months?"
            )
            correct = f"${semi_c:.2f} every six months (calculated as half of the ${annual_c} total annual contractual coupon payment)"
            d1 = f"${annual_c:.2f} every six months (calculated as the full stated annual coupon delivered on each payment date)"
            d2 = f"${round(semi_c * 0.75, 2):.2f} every six months (calculated by deducting the standard 25 percent tax withholding deduction)"
            d3 = f"${round(annual_c * 2.0, 2):.2f} every six months (calculated by doubling the annual yield to reflect semi-annual compounding)"

        s = build_sample(q, correct, [d1, d2, d3], "finance_bond_duration", rng)
        if s:
            samples.append(s)
    return samples


def generate_capital_budgeting_npv(rng, n=550):
    """Generate NPV, IRR, and Capital Budgeting Decision questions."""
    samples = []
    rates = [8, 10, 12, 14]

    for _ in range(n):
        r = rng.choice(rates)
        i0 = rng.choice([100, 150, 200, 250])  # thousands
        cf = rng.choice([35, 45, 55, 65, 75])
        years = rng.choice([3, 4, 5])

        # NPV = sum(cf / (1+r)^t) - i0
        pv_cfs = sum(cf / ((1.0 + r / 100.0) ** t) for t in range(1, years + 1))
        npv = round(pv_cfs - i0, 2)
        decision = "Accept the project" if npv > 0 else "Reject the project"

        # Distractor 1: undiscounted sum
        d1_val = round((cf * years) - i0, 2)
        # Distractor 2: forgot initial investment
        d2_val = round(pv_cfs, 2)
        # Distractor 3: opposite decision
        opp_decision = "Reject the project" if npv > 0 else "Accept the project"

        q = (
            f"An industrial capital expenditure requires an initial investment of ${i0},000 at t = 0 and generates "
            f"equal cash inflows of ${cf},000 per year for {years} years. If the company's cost of capital is {r}%, "
            f"what is the Net Present Value (NPV) and the resulting capital budgeting investment decision?"
        )

        correct = f"NPV is ${npv:.2f} thousand; {decision} (NPV is {'positive' if npv > 0 else 'negative'}, creating {'shareholder value' if npv > 0 else 'economic loss'})"
        d1 = f"NPV is ${d1_val:.2f} thousand; Accept the project (erroneously summing nominal un-discounted annual cash flows)"
        d2 = f"NPV is ${d2_val:.2f} thousand; Accept the project (erroneously omitting deduction of the initial capital outlay)"
        d3 = f"NPV is ${npv:.2f} thousand; {opp_decision} (erroneously reversing standard capital allocation net present value criteria)"

        s = build_sample(q, correct, [d1, d2, d3], "finance_capital_budgeting", rng)
        if s:
            samples.append(s)
    return samples


def generate_valuation_fcf_gordon(rng, n=550):
    """Generate Gordon Growth Model and Free Cash Flow valuation questions."""
    samples = []
    for _ in range(n):
        d0 = round(rng.uniform(1.5, 4.0), 2)
        g = rng.choice([2.0, 3.0, 4.0, 5.0])
        r = rng.choice([7.0, 8.0, 9.0, 10.0, 11.0])

        d1 = round(d0 * (1.0 + g / 100.0), 3)
        price = round(d1 / ((r - g) / 100.0), 2)

        # Distractor 1: used D0 instead of D1
        d1_price = round(d0 / ((r - g) / 100.0), 2)
        # Distractor 2: added g in denominator instead of subtracting (r + g)
        d2_price = round(d1 / ((r + g) / 100.0), 2)
        # Distractor 3: omitted growth rate in denominator (used r only)
        d3_price = round(d1 / (r / 100.0), 2)

        q = (
            f"A dividend-paying stock just paid an annual dividend of ${d0:.2f} per share (D0). "
            f"Dividends are projected to grow indefinitely at a constant rate of {g}% per year. "
            f"If the required rate of return on equity is {r}%, what is the intrinsic value per share under the Gordon Growth Model?"
        )

        correct = f"${price:.2f} per share (calculated as next year's dividend D1 of ${d1:.2f} divided by (r - g) of {r-g:.1f}%)"
        d1 = f"${d1_price:.2f} per share (erroneously capitalizing current dividend D0 without incorporating next period growth)"
        d2 = f"${d2_price:.2f} per share (erroneously adding the growth rate to the required return in the valuation denominator)"
        d3 = f"${d3_price:.2f} per share (erroneously treating the dividend as a flat perpetuity with zero perpetual growth)"

        s = build_sample(q, correct, [d1, d2, d3], "finance_valuation_gordon", rng)
        if s:
            samples.append(s)
    return samples


def generate_options_put_call(rng, n=550):
    """Generate Put-Call Parity and Option Payoff questions."""
    samples = []
    for _ in range(n):
        s0 = rng.choice([50, 75, 100, 125, 150])
        strike = s0
        c_price = round(s0 * rng.uniform(0.06, 0.12), 2)
        r = 4.0
        t = 1.0
        # Discounted strike = K * e^(-rT) approx K / (1 + r)
        pv_k = round(strike / (1.0 + r / 100.0), 2)
        # Put-Call Parity: C + PV(K) = P + S0 => P = C + PV(K) - S0
        put_price = round(c_price + pv_k - s0, 2)

        # Distractor 1: forgot discounting strike (P = C + K - S0)
        d1_put = round(c_price + strike - s0, 2)
        # Distractor 2: swapped signs (P = S0 + PV(K) - C)
        d2_put = round(s0 + pv_k - c_price, 2)
        # Distractor 3: parity inversion (P = C - PV(K) + S0)
        d3_put = round(c_price - pv_k + s0, 2)

        q = (
            f"A non-dividend-paying stock currently trades at ${s0:.2f}. A 1-year European call option with a strike price "
            f"of ${strike:.2f} trades at ${c_price:.2f}. If the annual risk-free interest rate is {r}% (PV of strike = ${pv_k:.2f}), "
            f"what is the theoretical arbitrage-free price of the corresponding 1-year European put option under Put-Call Parity?"
        )

        correct = f"${put_price:.2f} (derived from Put-Call Parity relationship: Put = Call + PV(Strike) - Stock Price)"
        d1 = f"${d1_put:.2f} (erroneously omitting present value discounting on the contractual strike price)"
        d2 = f"${d2_put:.2f} (erroneously reversing underlying asset and long derivative positions in parity)"
        d3 = f"${d3_put:.2f} (erroneously subtracting strike present value from call premium while adding equity)"

        s = build_sample(q, correct, [d1, d2, d3], "finance_options_parity", rng)
        if s:
            samples.append(s)
    return samples


def generate_liquidity_working_capital(rng, n=550):
    """Generate Cash Conversion Cycle and Quick Ratio questions."""
    samples = []
    for _ in range(n):
        dio = rng.randint(40, 80)
        dso = rng.randint(30, 60)
        dpo = rng.randint(25, 55)

        ccc = dio + dso - dpo

        # Distractor 1: added DPO instead of subtracting
        d1_ccc = dio + dso + dpo
        # Distractor 2: omitted DPO entirely
        d2_ccc = dio + dso
        # Distractor 3: subtracted DSO instead of DPO
        d3_ccc = dio - dso + dpo

        q = (
            f"A retail corporation reports Days Inventory Outstanding (DIO) of {dio} days, Days Sales Outstanding (DSO) "
            f"of {dso} days, and Days Payables Outstanding (DPO) of {dpo} days. "
            f"What is the company's Cash Conversion Cycle (CCC)?"
        )

        correct = f"{ccc} days (calculated as Days Inventory plus Days Sales Outstanding minus Days Payables Outstanding)"
        d1 = f"{d1_ccc} days (erroneously adding supplier payables deferral time instead of subtracting payables credit)"
        d2 = f"{d2_ccc} days (erroneously omitting the trade credit financing buffer provided by accounts payable terms)"
        d3 = f"{d3_ccc} days (erroneously subtracting customer receivable collection time from operational inventory cycle)"

        s = build_sample(q, correct, [d1, d2, d3], "finance_working_capital", rng)
        if s:
            samples.append(s)
    return samples


def main():
    print("================================================================")
    print(" GENERATING EXPANDED GROUNDED FINANCE DATASET (TRACK I)")
    print("================================================================")

    rng = random.Random(42)

    generated = []
    generated.extend(generate_wacc_capm(rng, n=600))
    generated.extend(generate_dupont_roe(rng, n=550))
    generated.extend(generate_bond_pricing_duration(rng, n=600))
    generated.extend(generate_capital_budgeting_npv(rng, n=550))
    generated.extend(generate_valuation_fcf_gordon(rng, n=550))
    generated.extend(generate_options_put_call(rng, n=550))
    generated.extend(generate_liquidity_working_capital(rng, n=550))

    print(f"Generated {len(generated)} grounded financial reasoning samples.")

    # Load existing finance samples if available
    existing = []
    if os.path.exists(TRACK_I_BASE_DIR):
        print(f"Loading existing finance samples from {TRACK_I_BASE_DIR}...")
        try:
            ds_base = load_from_disk(TRACK_I_BASE_DIR)
            for split in ["train", "test"]:
                if split in ds_base:
                    for ex in ds_base[split]:
                        existing.append({
                            "question": ex["question"].strip(),
                            "choices": [str(c).strip() for c in ex["choices"]],
                            "text": ex["text"].strip(),
                            "label": int(ex["label"]),
                            "source": ex.get("source", "finance_base"),
                        })
            print(f"Loaded {len(existing)} existing finance samples.")
        except Exception as e:
            print(f"Warning: could not load base finance: {e}")

    all_samples = existing + generated
    print(f"\nTotal combined finance samples before deduplication: {len(all_samples)}")

    seen = set()
    deduped = []
    for s in all_samples:
        norm_q = " ".join(s["question"].strip().lower().split())
        if norm_q not in seen:
            seen.add(norm_q)
            deduped.append(s)

    print(f"Unique finance questions after deduplication: {len(deduped)}")
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
    print(f">>> Successfully saved expanded finance dataset to {OUTPUT_DIR}")

    # Mirror to scratch/worktrees/track-i/data/finance_expanded
    track_i_target = "./scratch/worktrees/track-i/data/finance_expanded"
    os.makedirs(track_i_target, exist_ok=True)
    ds_dict.save_to_disk(track_i_target)
    print(f">>> Successfully mirrored expanded dataset to {track_i_target}")


if __name__ == "__main__":
    main()
