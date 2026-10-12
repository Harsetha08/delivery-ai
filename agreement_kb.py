"""Merchant-agreement knowledge base (v2).
 - Base agreements + dated amendments per merchant (SQLite); amendment clauses replace same-numbered base clauses
 - Terms are extracted ONLY from clauses about that topic (a '14 days' in a payment clause is not a dispute window)
 - Exceptions ('except', 'thereafter', 'business days'...) and conflicting values are FLAGGED, never silently resolved
 - Drafts stay hidden until a person approves; the bot only sees approved text in force on the date asked
 - compare(): what changed between two dates; search(): clause retrieval with domain-synonym expansion"""
import re, sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from support_bot import parse_clauses, SupportBot, AGREEMENT

SCHEMA = """
CREATE TABLE IF NOT EXISTS agreements(id INTEGER PRIMARY KEY, merchant_id TEXT, doc_type TEXT,
  effective_from TEXT, source TEXT, status TEXT DEFAULT 'draft', approved_by TEXT);
CREATE TABLE IF NOT EXISTS clauses(agreement_id INT, number INT, title TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS terms(agreement_id INT, key TEXT, value REAL, clause_number INT, flag TEXT DEFAULT '');
"""
# term -> (value regex, regex the CLAUSE must match to be about this topic)
PATTERNS = {
    "commission_pct": (r"(?:commission|fee)\D{0,40}?(\d+(?:\.\d+)?)\s?%", r"commission|\bfee"),
    "prep_sla_min": (r"within (\d+) minutes", r"prepar|ready"),
    "late_grace_min": (r"(?:by more than|beyond|over) (\d+) minutes", r"\blate\b|exceed"),
    "late_penalty_pct": (r"(?:penalty|deduction|charge)\D{0,30}?(\d+(?:\.\d+)?)\s?%", r"\blate\b|exceed"),
    "dispute_window_days": (r"within (\d+) (?:calendar )?days", r"dispute|claim"),
    "manager_review_usd": (r"(?:above|over|exceeding) (\d+) USD", r"review|escalat|manager|approval"),
    "payment_days": (r"(?:paid|pays?|payment)\D{0,25}?within (\d+) days", r"payment|paid|pays|settle"),
}
EXC = re.compile(r"\b(except|unless|provided that|notwithstanding|subject to|force majeure|public holidays?|holidays?|"
                 r"business days?|thereafter|tiered|reduced to|increased to)\b", re.I)
REQUIRED = ["commission_pct", "prep_sla_min", "late_grace_min", "late_penalty_pct", "dispute_window_days"]
RULE_KEYS = ["prep_sla_min", "late_grace_min", "late_penalty_pct", "dispute_window_days", "manager_review_usd"]
DEFAULTS = {"manager_review_usd": 100.0}   # platform policy used when a contract is silent
HIGHER_IS_BETTER_FOR_MERCHANT = {"commission_pct": False, "prep_sla_min": True, "late_grace_min": True,
    "late_penalty_pct": False, "dispute_window_days": True, "manager_review_usd": True, "payment_days": False}
SYN = {"fee": "commission", "fees": "commission", "cut": "commission", "take": "commission", "charge": "commission",
       "percentage": "commission", "slow": "late preparation penalty", "kitchen": "preparation", "charged": "penalty",
       "complain": "dispute", "complaint": "quality complaints", "complaints": "quality complaints",
       "deadline": "dispute window days", "wrong": "incorrect", "bag": "items missing packing", "refunds": "refund",
       "refund": "refunds partner-funded", "senior": "manager review", "approval": "manager review",
       "payout": "payment settlement", "photo": "photo evidence"}

def _tok(s): return [w.rstrip("s") for w in re.findall(r"[a-z]+", s.lower()) if w not in ENGLISH_STOP_WORDS]
def expand_query(q): return q + " " + " ".join(SYN[w] for w in re.findall(r"[a-z]+", q.lower()) if w in SYN)

def load_text(path):
    p = path.lower()
    if p.endswith((".txt", ".md")): return open(path, encoding="utf-8").read()
    if p.endswith(".docx"):
        import docx; return "\n".join(x.text for x in docx.Document(path).paragraphs)
    if p.endswith(".pdf"):
        import pypdf; return "\n".join(pg.extract_text() or "" for pg in pypdf.PdfReader(path).pages)
    raise ValueError("Unsupported file type")

class AgreementKB:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False); self.db.executescript(SCHEMA)

    def ingest(self, merchant_id, text, doc_type="base", effective_from="2026-01-01", source="inline"):
        """Parse and store as DRAFT. Returns (agreement_id, warnings a reviewer must clear)."""
        clauses = parse_clauses(text)
        warnings = [] if clauses else ["No numbered clauses found; check formatting"]
        aid = self.db.execute("INSERT INTO agreements(merchant_id,doc_type,effective_from,source) VALUES(?,?,?,?)",
                              (merchant_id, doc_type, effective_from, source)).lastrowid
        seen = {}
        for c in clauses:
            n = int(c["id"]); blob = f"{c['title']} {c['text']}"
            self.db.execute("INSERT INTO clauses VALUES(?,?,?,?)", (aid, n, c["title"], c["text"]))
            exc = EXC.search(c["text"])
            for key, (pat, gate) in PATTERNS.items():
                m = re.search(pat, c["text"], re.I) if re.search(gate, blob, re.I) else None
                if not m: continue
                val, flag = float(m.group(1)), (f"exception: '{exc.group(0)}'" if exc else "")
                self.db.execute("INSERT INTO terms VALUES(?,?,?,?,?)", (aid, key, val, n, flag))
                seen.setdefault(key, []).append((val, n))
                if flag: warnings.append(f"Clause {n} ({key}): {flag}; a person must read this before approval")
        for key, vals in seen.items():
            if len({v for v, _ in vals}) > 1:
                cl = ", ".join(str(n) for _, n in vals)
                self.db.execute("UPDATE terms SET flag=? WHERE agreement_id=? AND key=?", (f"conflict between clauses {cl}", aid, key))
                warnings.append(f"{key} has different values in clauses {cl}")
        if doc_type == "base":
            warnings += [f"Missing expected term: {k}" for k in REQUIRED if k not in seen]
        self.db.commit()
        return aid, warnings

    def review(self, aid):
        return self.db.execute("""SELECT t.key, t.value, t.clause_number, t.flag, c.text FROM terms t
            JOIN clauses c ON c.agreement_id=t.agreement_id AND c.number=t.clause_number
            WHERE t.agreement_id=? ORDER BY t.clause_number""", (aid,)).fetchall()

    def approve(self, aid, reviewer):
        self.db.execute("UPDATE agreements SET status='approved', approved_by=? WHERE id=?", (reviewer, aid)); self.db.commit()

    def effective_clauses(self, merchant_id, as_of):
        docs = self.db.execute("SELECT id FROM agreements WHERE merchant_id=? AND status='approved' AND effective_from<=? "
                               "ORDER BY effective_from, id", (merchant_id, as_of)).fetchall()
        merged = {}
        for (aid,) in docs:
            for n, title, text in self.db.execute("SELECT number,title,text FROM clauses WHERE agreement_id=?", (aid,)):
                merged[n] = {"id": n, "title": title, "text": text, "agreement_id": aid}
        return [merged[n] for n in sorted(merged)]

    def terms_as_of(self, merchant_id, as_of):
        out = {}
        for c in self.effective_clauses(merchant_id, as_of):
            for key, val, flag in self.db.execute("SELECT key,value,flag FROM terms WHERE agreement_id=? AND clause_number=?",
                                                  (c["agreement_id"], c["id"])):
                if key in out and out[key]["value"] != val:
                    flag = f"conflict between clauses {out[key]['clause']}, {c['id']}"
                    out[key]["flag"] = flag
                out[key] = {"value": val, "clause": c["id"], "agreement_id": c["agreement_id"], "flag": flag or out.get(key, {}).get("flag", "")}
        return out

    def flagged(self, merchant_id, as_of, keys=None):
        """[(term, clause, reason)] for terms that carry an exception or conflict."""
        return [(k, v["clause"], v["flag"]) for k, v in self.terms_as_of(merchant_id, as_of).items()
                if v["flag"] and (keys is None or k in keys)]

    def compare(self, merchant_id, date_a, date_b):
        """What changed between two dates: term changes (with effect on the merchant) and reworded clauses."""
        ta, tb = self.terms_as_of(merchant_id, date_a), self.terms_as_of(merchant_id, date_b)
        changes = []
        for k in sorted(set(ta) | set(tb)):
            old, new = ta.get(k, {}).get("value"), tb.get(k, {}).get("value")
            if old == new: continue
            up = HIGHER_IS_BETTER_FOR_MERCHANT.get(k)
            effect = "neutral" if old is None or new is None or up is None else ("better" if (new > old) == up else "worse")
            changes.append({"term": k, "old": old, "new": new, "clause": tb.get(k, ta.get(k))["clause"], "effect": effect})
        ca = {c["id"]: c["text"] for c in self.effective_clauses(merchant_id, date_a)}
        cb = {c["id"]: c["text"] for c in self.effective_clauses(merchant_id, date_b)}
        reworded = sorted(n for n in set(ca) | set(cb) if ca.get(n) != cb.get(n) and n not in {c["clause"] for c in changes})
        return {"term_changes": changes, "other_clauses_changed": reworded}

    def search(self, merchant_id, query, as_of, k=3, expand=True):
        cl = self.effective_clauses(merchant_id, as_of)
        if not cl: return []
        docs = [f"{c['title']} {c['text']}" for c in cl]
        v = TfidfVectorizer(tokenizer=_tok, lowercase=False, token_pattern=None).fit(docs)
        s = cosine_similarity(v.transform([expand_query(query) if expand else query]), v.transform(docs))[0]
        return [cl[i] for i in s.argsort()[::-1][:k] if s[i] > 0]

    def bot_for(self, merchant_id, as_of):
        """Rule engine built from the approved terms in force on that date, citing that merchant's real clause numbers."""
        cl = self.effective_clauses(merchant_id, as_of)
        if not cl: raise LookupError(f"No approved agreement for {merchant_id} on {as_of}")
        t = self.terms_as_of(merchant_id, as_of)
        terms = {k: v["value"] for k, v in t.items()}
        for k, dv in DEFAULTS.items(): terms.setdefault(k, dv)
        missing = [k for k in RULE_KEYS if k not in terms]
        if missing: raise LookupError(f"Agreement lacks terms the rules need: {missing}")
        bot = SupportBot("\n".join(f"{c['id']}. {c['title']}. {c['text']}" for c in cl))
        bot.terms = {**{k: None for k in PATTERNS}, **terms}
        find = lambda rx: next((str(c["id"]) for c in cl if re.search(rx, f"{c['title']} {c['text']}", re.I)), None)
        bot.clause_of = {k: str(v["clause"]) for k, v in t.items()}
        bot.clause_of.update(missing_item=find(r"missing|incorrect"), quality=find(r"quality|photo"))
        return bot

if __name__ == "__main__":
    kb = AgreementKB(); a, w = kb.ingest("M1", AGREEMENT); kb.approve(a, "legal")
    print("warnings:", w); [print(" ", r[:4]) for r in kb.review(a)]
