"""
HawkerSense v2.0 — Demand Forecasting & Decision Engine
for Informal Street Vendors (Pakistan / South Asia)

A probabilistic decision engine, not just a predictor.
Uses Monte Carlo simulation for uncertainty quantification.
Includes adaptive learning from daily waste feedback.

Author: HawkerSense Research Project
License: MIT
"""

import random
import math
import json
import os
import datetime
from collections import defaultdict


# ─────────────────────────────────────────────
# PRODUCT KNOWLEDGE BASE
# ─────────────────────────────────────────────

BASE_DEMAND = {
    "samosa":   70,
    "chai":     90,
    "paratha":  55,
    "gola":     40,   # ice candy
    "chana":    60,   # chana chaat
}

SALE_PRICE = {
    "samosa": 30,
    "chai":   25,
    "paratha": 60,
    "gola":   20,
    "chana":  40,
}

COST_PRICE = {
    "samosa": 14,
    "chai":   10,
    "paratha": 28,
    "gola":   8,
    "chana":  18,
}

DAY_MULTIPLIER = {
    "monday":    0.80,
    "tuesday":   0.82,
    "wednesday": 0.85,
    "thursday":  0.88,
    "friday":    1.10,
    "saturday":  1.20,
    "sunday":    1.15,
}

# Weather effect is product-specific (not one-size-fits-all)
WEATHER_MULTIPLIER = {
    "samosa": {"hot": 0.85, "normal": 1.00, "rainy": 1.15, "cold": 1.10},
    "chai":   {"hot": 0.80, "normal": 1.00, "rainy": 1.20, "cold": 1.35},
    "paratha":{"hot": 0.90, "normal": 1.00, "rainy": 1.05, "cold": 1.15},
    "gola":   {"hot": 1.60, "normal": 1.00, "rainy": 0.40, "cold": 0.20},
    "chana":  {"hot": 0.90, "normal": 1.00, "rainy": 0.85, "cold": 0.95},
}

EVENT_MULTIPLIER = {
    "none":          1.00,
    "weekly bazaar": 1.30,
    "cricket match": 1.45,
    "wedding nearby":1.25,
    "eid/festival":  1.80,
}


# ─────────────────────────────────────────────
# MONTE CARLO ENGINE
# ─────────────────────────────────────────────

def compute_base_demand(product, day, weather, event):
    """Compute the expected base demand from contextual factors."""
    base = BASE_DEMAND.get(product, 60)
    dm = DAY_MULTIPLIER.get(day.lower(), 1.0)
    wm = WEATHER_MULTIPLIER.get(product, {}).get(weather.lower(), 1.0)
    em = EVENT_MULTIPLIER.get(event.lower(), 1.0)
    return base * dm * wm * em


def monte_carlo_simulation(base_demand, n_simulations=500, noise_factor=0.15):
    """
    Run N Monte Carlo simulations to model demand uncertainty.

    Each run adds:
      - proportional noise (scaled to base)
      - random day-level jitter (±8 units)

    Returns a list of N integer demand outcomes.
    """
    results = []
    for _ in range(n_simulations):
        noise = base_demand * noise_factor * (random.random() * 2 - 1)
        jitter = (random.random() - 0.5) * 8
        simulated = max(5, round(base_demand + noise + jitter))
        results.append(simulated)
    return results


def percentile(data, p):
    """Return the p-th percentile of a list."""
    sorted_data = sorted(data)
    idx = int(math.floor((p / 100.0) * (len(sorted_data) - 1)))
    return sorted_data[idx]


# ─────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────

def compute_decision(product, day, weather, event,
                     sale_price=None, cost_price=None,
                     waste_yesterday=0, n_simulations=500):
    """
    The core decision function.

    Returns a structured decision dict with:
      - demand forecast (range + median)
      - recommended stock
      - unsold risk %
      - profit estimate
      - confidence score
      - late-hour pricing suggestion
      - textual recommendation
    """
    # Use defaults if prices not provided
    sale_price = sale_price or SALE_PRICE.get(product, 30)
    cost_price = cost_price or COST_PRICE.get(product, 14)

    # Adaptive learning: reduce base demand if vendor wasted stock yesterday
    waste_adjustment = round(waste_yesterday * 0.6)

    base = compute_base_demand(product, day, weather, event)
    adjusted_base = max(10, base - waste_adjustment)

    # Monte Carlo
    simulations = monte_carlo_simulation(adjusted_base, n_simulations)

    p10 = percentile(simulations, 10)
    p50 = percentile(simulations, 50)   # median
    p90 = percentile(simulations, 90)

    # Stock recommendation: slightly below median to minimise waste
    stock_low  = round(p50 * 0.92)
    stock_mid  = round(p50 * 0.97)
    stock_high = round(p50 * 1.05)

    # Unsold risk: proportion of simulations where demand < recommended stock
    unsold_risk_pct = round(
        sum(1 for s in simulations if s < stock_mid) / len(simulations) * 100
    )

    # Confidence: penalised for high weather/event multipliers (more uncertainty)
    wm = WEATHER_MULTIPLIER.get(product, {}).get(weather.lower(), 1.0)
    em = EVENT_MULTIPLIER.get(event.lower(), 1.0)
    dm = DAY_MULTIPLIER.get(day.lower(), 1.0)
    confidence = min(95, round(85 - abs(dm - 1) * 20 - abs(wm - 1) * 15 - abs(em - 1) * 8))

    # Profit estimate
    estimated_profit = round(stock_mid * (sale_price - cost_price))

    # Late-hour pricing (clear remaining stock)
    late_price = round(sale_price * 0.88) if sale_price > 30 else sale_price - 3

    # Risk bucket
    if unsold_risk_pct < 10:
        risk_level = "LOW"
    elif unsold_risk_pct < 20:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # Natural language recommendation
    notes = []
    if event.lower() != "none":
        boost = round((em - 1) * 100)
        notes.append(f"Event nearby ({event}) boosts demand by ~{boost}%.")
    if weather.lower() == "hot" and product in ("chai", "samosa"):
        notes.append("Hot weather suppresses demand for this item — stock conservatively.")
    if weather.lower() == "hot" and product == "gola":
        notes.append("Hot weather — strong demand for cold items. Stock the higher end.")
    if waste_adjustment > 0:
        notes.append(f"Learning applied: reduced forecast by {waste_adjustment} units based on yesterday's waste.")

    recommendation = (
        f"Stock {stock_low}–{stock_high} {product.title()}. "
        f"Risk of unsold items: {unsold_risk_pct}%. "
        f"After 7pm, consider pricing at Rs. {late_price} to clear remaining stock. "
        + " ".join(notes)
    )

    return {
        "product": product,
        "day": day,
        "weather": weather,
        "event": event,
        "forecast": {
            "p10": p10,
            "median": p50,
            "p90": p90,
        },
        "stock_recommendation": {
            "low": stock_low,
            "mid": stock_mid,
            "high": stock_high,
        },
        "unsold_risk_pct": unsold_risk_pct,
        "risk_level": risk_level,
        "confidence_pct": confidence,
        "estimated_profit_rs": estimated_profit,
        "late_hour_price_rs": late_price,
        "waste_adjustment_applied": waste_adjustment,
        "recommendation": recommendation,
    }


# ─────────────────────────────────────────────
# ADAPTIVE LEARNING LOG
# ─────────────────────────────────────────────

LOG_FILE = "hawkersense_log.json"


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []


def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def log_run(decision, waste_actual=None):
    """Append a run to the adaptive learning log."""
    log = load_log()
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "product": decision["product"],
        "day": decision["day"],
        "weather": decision["weather"],
        "event": decision["event"],
        "recommended_stock": decision["stock_recommendation"]["mid"],
        "forecast_median": decision["forecast"]["median"],
        "unsold_risk_pct": decision["unsold_risk_pct"],
        "waste_actual": waste_actual,
    }
    log.append(entry)
    save_log(log)


# ─────────────────────────────────────────────
# TERMINAL REPORT
# ─────────────────────────────────────────────

def print_report(d):
    """Pretty-print a decision report to terminal."""
    line = "─" * 52
    print(f"\n{'═'*52}")
    print(f"  HAWKERSENSE DECISION ENGINE  v2.0")
    print(f"{'═'*52}")
    print(f"  Product : {d['product'].title()}")
    print(f"  Day     : {d['day'].title()}")
    print(f"  Weather : {d['weather'].title()}")
    print(f"  Event   : {d['event'].title()}")
    print(line)
    print(f"  FORECAST (Monte Carlo, 500 runs)")
    print(f"    10th pct : {d['forecast']['p10']} units")
    print(f"    Median   : {d['forecast']['median']} units")
    print(f"    90th pct : {d['forecast']['p90']} units")
    print(line)
    print(f"  DECISION")
    print(f"    Recommended stock : {d['stock_recommendation']['low']}–{d['stock_recommendation']['high']} units")
    print(f"    Unsold risk       : {d['unsold_risk_pct']}%  [{d['risk_level']}]")
    print(f"    Confidence        : {d['confidence_pct']}%")
    print(f"    Est. profit       : Rs. {d['estimated_profit_rs']:,}")
    print(f"    Late-hour price   : Rs. {d['late_hour_price_rs']}")
    if d['waste_adjustment_applied'] > 0:
        print(f"    Learning adj.     : -{d['waste_adjustment_applied']} units (from yesterday's waste)")
    print(line)
    print(f"  RECOMMENDATION")
    # Word-wrap at 50 chars
    words = d['recommendation'].split()
    line_buf = "    "
    for w in words:
        if len(line_buf) + len(w) > 52:
            print(line_buf)
            line_buf = "    " + w + " "
        else:
            line_buf += w + " "
    if line_buf.strip():
        print(line_buf)
    print(f"{'═'*52}\n")


# ─────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────

def prompt_choice(prompt, choices):
    choices_lower = [c.lower() for c in choices]
    while True:
        print(f"\n{prompt}")
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        val = input("Enter number or type: ").strip().lower()
        if val.isdigit() and 1 <= int(val) <= len(choices):
            return choices_lower[int(val) - 1]
        if val in choices_lower:
            return val
        print("  Invalid choice, try again.")


def run_cli():
    """Full interactive CLI for HawkerSense."""
    print("\n" + "═"*52)
    print("  Welcome to HawkerSense v2.0")
    print("  Demand Decision Engine for Street Vendors")
    print("═"*52)

    product = prompt_choice("Select product:", list(BASE_DEMAND.keys()))
    day = prompt_choice("Select day:", list(DAY_MULTIPLIER.keys()))
    weather = prompt_choice("Select weather:", ["hot", "normal", "rainy", "cold"])
    event = prompt_choice("Nearby event?", list(EVENT_MULTIPLIER.keys()))

    print(f"\n  Sale price per unit (default Rs. {SALE_PRICE[product]}): ", end="")
    sp_raw = input().strip()
    sale_price = int(sp_raw) if sp_raw.isdigit() else SALE_PRICE[product]

    print(f"  Cost price per unit (default Rs. {COST_PRICE[product]}): ", end="")
    cp_raw = input().strip()
    cost_price = int(cp_raw) if cp_raw.isdigit() else COST_PRICE[product]

    print("  Units wasted yesterday (0 if first run): ", end="")
    waste_raw = input().strip()
    waste_yesterday = int(waste_raw) if waste_raw.isdigit() else 0

    decision = compute_decision(
        product=product,
        day=day,
        weather=weather,
        event=event,
        sale_price=sale_price,
        cost_price=cost_price,
        waste_yesterday=waste_yesterday,
    )

    print_report(decision)
    log_run(decision, waste_actual=waste_yesterday)

    # Save JSON report
    report_file = f"hawkersense_report_{datetime.date.today()}.json"
    with open(report_file, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"  Report saved → {report_file}")
    print(f"  Learning log → {LOG_FILE}\n")


if __name__ == "__main__":
    run_cli()
