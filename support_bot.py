"""Partner support bot: parse merchant agreement -> structured terms + searchable
clauses -> resolve disputes with deterministic rules, cite clauses, escalate when unsure.
The LLM (optional hook) only drafts wording; it never decides money."""
import re
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

AGREEMENT = """
1. Commission. Platform charges a commission of 18% on food subtotal.
2. Preparation SLA. Partner shall have orders ready within 20 minutes of acceptance.
3. Late Preparation. If preparation exceeds the SLA by more than 10 minutes, a penalty of 5% of subtotal applies.
4. Missing or Incorrect Items. Partner bears the cost of items missing or incorrect due to packing error, up to the item value.
5. Quality Complaints. Refunds for quality complaints are partner-funded only with photo evidence submitted within 24 hours.
6. Dispute Window. Disputes must be raised within 7 days of the order date.
7. Escalation. Claims above 100 USD require review by a partner manager.
"""

def parse_clauses(text):
    return [{"id": m.group(1), "title": m.group(2), "text": m.group(3).strip()}
            for m in re.finditer(r"^(\d+)\.\s+([^.]+)\.\s+(.*)$", text, re.M)]

def extract_terms(text):
    g = lambda p, d=None: (lambda m: float(m.group(1)) if m else d)(re.search(p, text))
    return {"commission_pct": g(r"commission of (\d+)%"),
            "prep_sla_min": g(r"within (\d+) minutes of acceptance"),
            "late_grace_min": g(r"exceeds the SLA by more than (\d+) minutes"),
            "late_penalty_pct": g(r"penalty of (\d+)% of subtotal"),
            "dispute_window_days": g(r"within (\d+) days"),
            "manager_review_usd": g(r"above (\d+) USD")}

class Retriever:
    def __init__(self, clauses):
        self.c = clauses
        self.v = TfidfVectorizer(stop_words="english").fit([f"{c['title']} {c['text']}" for c in clauses])
        self.X = self.v.transform([f"{c['title']} {c['text']}" for c in clauses])
    def top(self, q, k=2):
        s = cosine_similarity(self.v.transform([q]), self.X)[0]
        return [self.c[i] for i in s.argsort()[::-1][:k] if s[i] > 0]

@dataclass
class Order:
    subtotal: float; accepted_to_ready_min: float; days_since_order: int
    photo_within_24h: bool = False; packing_error: bool = False; item_value: float = 0.0

@dataclass
class Decision:
    outcome: str; amount: float; reasons: list = field(default_factory=list); cites: list = field(default_factory=list)

def classify(msg):
    m = msg.lower()
    if any(w in m for w in ("late", "slow", "wait", "delay")): return "late_prep"
    if any(w in m for w in ("missing", "wrong item", "incorrect")): return "missing_item"
    if any(w in m for w in ("cold", "quality", "spilled", "stale")): return "quality"
    return "unknown"

class SupportBot:
    def __init__(self, agreement_text):
        self.clauses = parse_clauses(agreement_text)
        self.terms = extract_terms(agreement_text)
        self.ret = Retriever(self.clauses)
        # Which clause backs each rule. Default matches the sample agreement; AgreementKB.bot_for overrides it per merchant.
        self.clause_of = {"dispute_window_days": "6", "prep_sla_min": "2", "late_grace_min": "3", "late_penalty_pct": "3",
                          "manager_review_usd": "7", "missing_item": "4", "quality": "5"}

    def _c(self, *keys):
        out = []
        for k in keys:
            c = self.clause_of.get(k)
            if c and c not in out: out.append(c)
        return out

    def resolve(self, msg, o: Order) -> Decision:
        t, kind = self.terms, classify(msg)
        cites = [c["id"] for c in self.ret.top(msg)]
        if o.days_since_order > t["dispute_window_days"]:
            return Decision("reject", 0, [f"Raised after {t['dispute_window_days']:.0f}-day window"], self._c("dispute_window_days"))
        if kind == "late_prep":
            over = o.accepted_to_ready_min - t["prep_sla_min"]
            if over > t["late_grace_min"]:
                amt = round(o.subtotal * t["late_penalty_pct"] / 100, 2)
                d = Decision("partner_penalised", amt, [f"Prep ran {over:.0f} min over SLA"], self._c("prep_sla_min", "late_grace_min", "late_penalty_pct"))
            else:
                d = Decision("platform_absorbs", 0, [f"Within SLA + {t['late_grace_min']:.0f} min grace"], self._c("prep_sla_min", "late_grace_min", "late_penalty_pct"))
        elif kind == "missing_item":
            if o.packing_error:
                d = Decision("partner_funded", min(o.item_value, o.item_value), ["Packing error confirmed"], self._c("missing_item"))
            else:
                d = Decision("needs_review", 0, ["No packing-error evidence"], self._c("missing_item"))
        elif kind == "quality":
            ok = o.photo_within_24h
            d = Decision("partner_funded" if ok else "platform_absorbs", o.subtotal if ok else 0,
                         ["Photo evidence provided" if ok else "No photo within 24h"], self._c("quality"))
        else:
            return Decision("needs_review", 0, ["Could not classify dispute"], cites)
        if d.amount > t["manager_review_usd"]:
            d.reasons.append("Amount over manager-review threshold"); d.outcome = "escalate_manager"; d.cites.extend(self._c("manager_review_usd"))
        return d

    def reply(self, d: Decision):  # replace body with an LLM call to polish tone; keep facts fixed
        return (f"Outcome: {d.outcome.replace('_',' ')} (${d.amount:.2f}). "
                f"Basis: {'; '.join(d.reasons)}. Agreement clauses: {', '.join(d.cites)}.")

if __name__ == "__main__":
    bot = SupportBot(AGREEMENT)
    print(bot.terms, "\n")
    print(bot.reply(bot.resolve("Customer says order was very late", Order(40, 35, 2))))
    print(bot.reply(bot.resolve("Item missing from bag", Order(30, 15, 1, packing_error=True, item_value=8))))
    print(bot.reply(bot.resolve("Food arrived cold", Order(25, 15, 9, photo_within_24h=True))))
    print(bot.reply(bot.resolve("Wrong item and spilled", Order(25, 15, 3))))
