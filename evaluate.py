"""Evaluation report:  python evaluate.py
 1. Delivery-time prediction   2. Agreement retrieval   3. Interpretation audit   4. Unseen-merchant validation"""
import numpy as np
from eta_model import ETAModel, make_synthetic_data
from agreement_kb import AgreementKB, PATTERNS
from support_bot import extract_terms
from assistant import MerchantSupportAssistant, StubLLM
from fixtures import build_world, SCENARIOS, UNSEEN_TEXT, RETRIEVAL_QUERIES, AMENDMENT
from support_bot import AGREEMENT

def truth(d): return np.maximum(d.prep_min, d.pickup_min) + d.dropoff_min

def prediction():
    print("\n=== 1. DELIVERY-TIME PREDICTION (synthetic data) ===")
    df = make_synthetic_data(40000, seed=3)
    new = df.restaurant_id >= 190                      # restaurants the model never sees in training
    seen = df[~new]; train, test = seen.iloc[:28000], seen.iloc[28000:]; cold = df[new]
    m = ETAModel().fit(train)
    def row(name, d):
        p, y = m.predict(d), truth(d)
        print(f"{name:<26}{len(d):>7}  MAE {np.abs(p.eta_p50 - y).mean():5.2f}  p90 coverage {100*(y <= p.eta_p90).mean():5.1f}%  "
              f"late vs p50 {100*(y > p.eta_p50).mean():4.1f}%  late vs p90 {100*(y > p.eta_p90).mean():4.1f}%")
    row("All (known restaurants)", test)
    row("Peak hours", test[test.hour.isin([12, 13, 19, 20, 21])])
    row("Raining", test[test.rain == 1])
    row("Busy kitchen (queue>=8)", test[test.kitchen_queue >= 8])
    row("NEW restaurants (cold)", cold)
    y = truth(test)
    base_mean = np.abs(y - truth(train).mean()).mean()
    pick = test.driver_to_rest_km / test.driver_speed_kmh * 60 + 2; drop = test.rest_to_cust_km / test.driver_speed_kmh * 60 + 3
    base_heur = np.abs(y - (np.maximum(test.rest_avg_prep_7d, pick) + drop)).mean()
    mae = np.abs(m.predict(test).eta_p50 - y).mean()
    print(f"Baselines: always-the-average MAE {base_mean:.2f} | distance + restaurant-history rule MAE {base_heur:.2f} | model MAE {mae:.2f}")
    print("Prep-time only MAE: %.2f min" % np.abs(m.predict(test).prep_p50 - test.prep_min).mean())

def retrieval():
    print("\n=== 2. AGREEMENT RETRIEVAL (12 paraphrased merchant questions) ===")
    kb = AgreementKB(); a, _ = kb.ingest("M", AGREEMENT); kb.approve(a, "r")
    for expand in (False, True):
        r1 = r3 = mrr = 0; misses = []
        for q, gold in RETRIEVAL_QUERIES:
            ids = [c["id"] for c in kb.search("M", q, "2026-10-04", k=7, expand=expand)]
            r1 += ids[:1] == [gold]; r3 += gold in ids[:3]; mrr += 1 / (ids.index(gold) + 1) if gold in ids else 0
            if gold not in ids[:3]: misses.append(q)
        n = len(RETRIEVAL_QUERIES)
        print(f"{'with synonym expansion' if expand else 'plain TF-IDF':<24} recall@1 {r1/n:.0%}  recall@3 {r3/n:.0%}  MRR {mrr/n:.2f}  missed@3: {misses or 'none'}")
    print("Caveat: the synonym list was written while looking at these questions, so the gain is optimistic. Re-test on real merchant queries.")

def audit():
    print("\n=== 3. INTERPRETATION AUDIT: old whole-text regex vs knowledge base ===")
    w = build_world(); kb = w["kb"]
    cases = [("M15", "dispute_window_days", "7 (payment clause says 14)"), ("M16", "manager_review_usd", "none stated (50 is a commission discount)"),
             ("M11", "prep_sla_min", "20, with a holiday exception"), ("M13", "late_penalty_pct", "5%, tiered ('thereafter')"),
             ("M12", "dispute_window_days", "conflict: 7 vs 14")]
    print(f"{'merchant':<10}{'term':<22}{'truth':<46}{'old regex':<12}{'knowledge base'}")
    for m, key, truth_ in cases:
        old = extract_terms(UNSEEN_TEXT[m]).get(key); t = kb.terms_as_of(m, "2026-08-15").get(key)
        new = "not stated" if t is None else f"{t['value']:g}" + (f"  FLAGGED ({t['flag'][:28]})" if t["flag"] else "")
        print(f"{m:<10}{key:<22}{truth_:<46}{str(old):<12}{new}")

def unseen():
    print("\n=== 4. UNSEEN-MERCHANT VALIDATION (9 orders, 7 differently worded agreements) ===")
    w = build_world(); bot = MerchantSupportAssistant(w["kb"], StubLLM(), w["orders"], w["preds"], today="2026-10-04")
    ok = unsafe = over = 0
    for oid, m, _, _, expect, msg in SCENARIOS:
        r = bot.handle(m, msg)
        got = (r["route"],) if r["route"] == "escalated" else (r["route"], r["decision"], r["amount"])
        if got == expect: ok += 1; tag = "correct"
        elif got[0] == "escalated": over += 1; tag = "over-escalated (safe)"
        else: unsafe += 1; tag = "WRONG AUTO-RESOLUTION"
        print(f"  {oid} {m:<4} expected {str(expect):<48} got {str(got):<48} {tag}")
    print(f"Correct {ok}/{len(SCENARIOS)} | safe over-escalations {over} | wrong auto-resolutions {unsafe}  (this last number must stay 0)")

if __name__ == "__main__":
    prediction(); retrieval(); audit(); unseen()
