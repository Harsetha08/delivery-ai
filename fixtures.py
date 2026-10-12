"""Shared test data: merchants, agreements, orders, predictions, labelled retrieval queries."""
from agreement_kb import AgreementKB
from integration import LoggedPrediction
from support_bot import AGREEMENT, Order

AMENDMENT = ("1. Commission. Platform charges a commission of 15% on food subtotal.\n"
             "2. Preparation SLA. Partner shall have orders ready within 25 minutes of acceptance.")

STD = {"commission": "1. Commission. Platform charges a commission of 18% on food subtotal.",
       "sla": "2. Preparation SLA. Partner shall have orders ready within 20 minutes of acceptance.",
       "late": "3. Late Preparation. If preparation exceeds the SLA by more than 10 minutes, a penalty of 5% of subtotal applies.",
       "dispute": "4. Dispute Window. Disputes must be raised within 7 days of the order date."}
def agreement(**over): return "\n".join(over.get(k, v) for k, v in STD.items())

# "Unseen" merchants: different wording, exceptions, conflicts, clause order.
UNSEEN_TEXT = {
 "M10": ("1. Service Fee. The Platform retains a service fee equal to 20% of the food subtotal.\n"
         "2. Ready Time. The Restaurant must have each order ready for pickup within 18 minutes of order acceptance.\n"
         "3. Late Orders. Where ready time is exceeded by more than 5 minutes, a deduction of 8% of subtotal applies.\n"
         "4. Missing Items. The Restaurant bears the cost of items missing due to packing error.\n"
         "5. Claims Period. Any dispute must be submitted within 3 days of the order date.\n"
         "6. Manager Approval. Claims over 50 USD need manager review."),
 "M11": agreement(sla="2. Preparation SLA. Partner shall have orders ready within 20 minutes of acceptance, except on public holidays when 30 minutes applies."),
 "M12": agreement() + "\n5. Claims. Claims must be filed within 14 days of the order date.",
 "M13": agreement(late="3. Late Preparation. If preparation exceeds the SLA by more than 10 minutes, a penalty of 5% of subtotal applies for the first 10 minutes and 10% thereafter."),
 "M14": agreement(dispute="4. Dispute Window. Disputes must be raised within 5 business days of the order date."),
 "M15": "\n".join([STD["commission"], "2. Payment. The Platform pays the Partner within 14 days of settlement.",
                   "3. Preparation SLA. Partner shall have orders ready within 20 minutes of acceptance.",
                   "4. Late Preparation. If preparation exceeds the SLA by more than 10 minutes, a penalty of 5% of subtotal applies.",
                   "5. Dispute Window. Disputes must be raised within 7 days of the order date."]),
 "M16": agreement(commission="1. Commission. Platform charges a commission of 18% on food subtotal, reduced to 15% for orders above 50 USD."),
}

# id, merchant, order(subtotal, actual prep min, days since order), logged prediction p90, expected result, message
SCENARIOS = [
 ("O101", "M10", Order(50, 30, 1), 25, ("auto_resolved", "partner_penalised", 4.0), "Order O101 was late"),
 ("O102", "M10", Order(50, 21, 1), 25, ("auto_resolved", "platform_absorbs", 0.0), "Order O102 was late"),
 ("O103", "M10", Order(50, 30, 6), 25, ("auto_resolved", "reject", 0.0), "Order O103 was late"),
 ("O104", "M11", Order(50, 33, 1), 28, ("escalated",), "Order O104 was late"),
 ("O105", "M12", Order(50, 33, 1), 28, ("escalated",), "Order O105 was late"),
 ("O106", "M13", Order(50, 33, 1), 28, ("escalated",), "Order O106 was late"),
 ("O107", "M14", Order(50, 33, 1), 28, ("escalated",), "Order O107 was late"),
 ("O108", "M15", Order(50, 33, 9), 28, ("auto_resolved", "reject", 0.0), "Order O108 was late"),
 ("O109", "M16", Order(50, 33, 2), 28, ("auto_resolved", "partner_penalised", 2.5), "Order O109 was late"),
]

# (query, gold clause in the sample agreement)
RETRIEVAL_QUERIES = [
 ("What percentage do you charge me on each order?", 1), ("how big is your cut", 1),
 ("How fast must the kitchen have the food ready?", 2), ("What happens if my restaurant is slow", 3),
 ("Do I get charged when orders are late?", 3), ("who pays when something is missing from the bag", 4),
 ("a customer sent a photo of cold food, who refunds?", 5), ("how long do I have to complain about an order", 6),
 ("deadline to raise a dispute", 6), ("when does a manager need to look at my claim", 7),
 ("what amount needs senior approval", 7), ("wrong item delivered", 4),
]

def build_world():
    kb, warnings = AgreementKB(), {}
    def add(m, text, doc="base", eff="2026-01-01"):
        a, w = kb.ingest(m, text, doc, eff); kb.approve(a, "reviewer"); warnings.setdefault(m, []).extend(w)
    add("M1", AGREEMENT); add("M1", AMENDMENT, "amendment", "2026-07-01"); add("M2", AGREEMENT)
    for m, t in UNSEEN_TEXT.items(): add(m, t)
    orders = {"O1": {"merchant_id": "M1", "order_date": "2026-03-10", "order": Order(40, 33, 2)},
              "O2": {"merchant_id": "M1", "order_date": "2026-03-10", "order": Order(40, 31, 3)},
              "O3": {"merchant_id": "M1", "order_date": "2026-03-01", "order": Order(40, 40, 9)},
              "O4": {"merchant_id": "M2", "order_date": "2026-03-10", "order": Order(40, 40, 2)},
              "O5": {"merchant_id": "M1", "order_date": "2026-09-10", "order": Order(40, 33, 2)}}
    preds = {"O1": LoggedPrediction("O1", 22, 28, 40, 48), "O2": LoggedPrediction("O2", 29, 32, 45, 50),
             "O5": LoggedPrediction("O5", 22, 28, 40, 48)}
    for oid, m, o, p90, _, _ in SCENARIOS:
        orders[oid] = {"merchant_id": m, "order_date": "2026-08-15", "order": o}
        preds[oid] = LoggedPrediction(oid, p90 - 4, p90, p90 + 15, p90 + 22)
    return {"kb": kb, "orders": orders, "preds": preds, "warnings": warnings}
