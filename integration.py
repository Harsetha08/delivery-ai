"""Connects ETA predictions to merchant-agreement terms.
 1) At order time: compare predicted prep time to the contract SLA (early warning).
 2) At dispute time: use the prediction logged at order time as evidence.
The contract still decides the outcome; predictions only confirm it or flag it for review."""
import pandas as pd
from dataclasses import dataclass
from eta_model import ETAModel, make_synthetic_data, FEATS
from support_bot import SupportBot, Order, Decision, AGREEMENT

@dataclass
class LoggedPrediction:
    order_id: str; prep_p50: float; prep_p90: float; eta_p50: float; eta_p90: float

class DeliveryContractService:
    def __init__(self, model: ETAModel, bot: SupportBot):
        self.model, self.bot, self.log = model, bot, {}   # swap self.log for a database table

    def predict_and_log(self, order_id, features: pd.DataFrame):
        p = self.model.predict(features).iloc[0]
        rec = LoggedPrediction(order_id, p.prep_p50, p.prep_p90, p.eta_p50, p.eta_p90)
        self.log[order_id] = rec
        return rec, self.sla_risk(rec)

    def sla_risk(self, rec):
        t = self.bot.terms
        limit = t["prep_sla_min"] + t["late_grace_min"]      # penalty only starts beyond this
        if rec.prep_p50 > t["prep_sla_min"]:
            return {"risk": "high", "note": f"Typical prep {rec.prep_p50:.0f} min already exceeds the {t['prep_sla_min']:.0f}-min SLA; "
                    "tell the customer the realistic ETA and do not rely on the SLA."}
        if rec.prep_p90 > limit:
            return {"risk": "medium", "note": f"1-in-10 chance prep passes {limit:.0f} min (penalty zone); warn the merchant."}
        return {"risk": "low", "note": "Prep expected within SLA."}

    def resolve_dispute(self, order_id, message, order: Order) -> dict:
        d = self.bot.resolve(message, order)
        rec = self.log.get(order_id)
        evidence = "No prediction logged for this order"
        if rec and d.outcome == "partner_penalised":
            actual = order.accepted_to_ready_min
            if actual > rec.prep_p90:
                evidence = (f"Actual prep {actual:.0f} min exceeded the p90 prediction ({rec.prep_p90:.0f}). "
                            "Slower than expected for this order and kitchen load, so the penalty is confirmed.")
            else:
                evidence = (f"Actual prep {actual:.0f} min was within the predicted range (p50 {rec.prep_p50:.0f}, "
                            f"p90 {rec.prep_p90:.0f}). The delay matches normal load, not an unusual failure; sending for human review.")
                d = Decision("needs_review", d.amount, d.reasons + ["Contract penalty applies but prediction shows normal load"], d.cites)
        elif rec:
            evidence = f"Prediction on file (prep p50 {rec.prep_p50:.0f}, p90 {rec.prep_p90:.0f}); no effect on this outcome."
        return {"decision": d, "prediction_evidence": evidence, "reply": self.bot.reply(d) + " Evidence: " + evidence}

if __name__ == "__main__":
    df = make_synthetic_data()
    svc = DeliveryContractService(ETAModel().fit(df.iloc[:30000]), SupportBot(AGREEMENT))
    test = df.iloc[30000:30003]
    for i, (_, row) in enumerate(test.iterrows()):
        oid = f"A{i}"
        rec, risk = svc.predict_and_log(oid, row[FEATS].to_frame().T.astype(float))
        print(f"{oid}: prep p50={rec.prep_p50:.1f} p90={rec.prep_p90:.1f} | risk={risk['risk']} - {risk['note']}")
    print()
    # Scenario 1: restaurant took far longer than predicted
    r = svc.log["A0"]; slow = r.prep_p90 + 12
    print(svc.resolve_dispute("A0", "Order was very late", Order(40, slow, 1))["reply"])
    # Scenario 2: prep was longer than the 30-min contract limit but within the predicted range
    r = svc.log["A1"]; normal = max(31, r.prep_p50)
    print(svc.resolve_dispute("A1", "Order was very late", Order(40, normal, 1))["reply"])
    # Scenario 3: no logged prediction
    print(svc.resolve_dispute("ZZ", "Order was very late", Order(40, 45, 1))["reply"])
