"""Merchant-agreement knowledge base.
 - Stores base agreements and dated amendments per merchant (SQLite)
 - Splits into clauses; amendment clauses replace base clauses with the same number
 - Extracts structured terms WITH the clause they came from
 - Terms stay 'draft' until a person approves them; the bot only sees approved agreements
 - Answers 'what applied to this merchant on this date?' and searches clauses"""
import re, sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from support_bot import parse_clauses, SupportBot, AGREEMENT

SCHEMA = """
CREATE TABLE IF NOT EXISTS agreements(id INTEGER PRIMARY KEY, merchant_id TEXT, doc_type TEXT,
  effective_from TEXT, source TEXT, status TEXT DEFAULT 'draft', approved_by TEXT);
CREATE TABLE IF NOT EXISTS clauses(agreement_id INT, number INT, title TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS terms(agreement_id INT, key TEXT, value REAL, clause_number INT);
"""
# term key -> regex (first group = number). Extend as you see real contract wording.
PATTERNS = {
    "commission_pct": r"commission of (\d+(?:\.\d+)?)%",
    "prep_sla_min": r"within (\d+) minutes of acceptance",
    "late_grace_min": r"exceeds the SLA by more than (\d+) minutes",
    "late_penalty_pct": r"penalty of (\d+(?:\.\d+)?)% of subtotal",
    "dispute_window_days": r"within (\d+) days",
    "manager_review_usd": r"above (\d+) USD",
    "payment_days": r"paid within (\d+) days of (?:delivery|settlement)",
}
REQUIRED = ["commission_pct", "prep_sla_min", "dispute_window_days"]

def load_text(path):
    """Read .txt/.md directly; .docx and .pdf if python-docx / pypdf are installed."""
    p = path.lower()
    if p.endswith((".txt", ".md")):
        return open(path, encoding="utf-8").read()
    if p.endswith(".docx"):
        import docx
        return "\n".join(x.text for x in docx.Document(path).paragraphs)
    if p.endswith(".pdf"):
        import pypdf
        return "\n".join(pg.extract_text() or "" for pg in pypdf.PdfReader(path).pages)
    raise ValueError("Unsupported file type")

class AgreementKB:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path); self.db.executescript(SCHEMA)

    def ingest(self, merchant_id, text, doc_type="base", effective_from="2026-01-01", source="inline"):
        """Parse and store as DRAFT. Returns (agreement_id, warnings)."""
        clauses = parse_clauses(text)
        warnings = [] if clauses else ["No numbered clauses found; check formatting"]
        cur = self.db.execute("INSERT INTO agreements(merchant_id,doc_type,effective_from,source) VALUES(?,?,?,?)",
                              (merchant_id, doc_type, effective_from, source))
        aid = cur.lastrowid
        found = set()
        for c in clauses:
            self.db.execute("INSERT INTO clauses VALUES(?,?,?,?)", (aid, int(c["id"]), c["title"], c["text"]))
            for key, pat in PATTERNS.items():
                m = re.search(pat, c["text"])
                if m:
                    self.db.execute("INSERT INTO terms VALUES(?,?,?,?)", (aid, key, float(m.group(1)), int(c["id"])))
                    found.add(key)
        if doc_type == "base":
            warnings += [f"Missing expected term: {k}" for k in REQUIRED if k not in found]
        self.db.commit()
        return aid, warnings

    def review(self, aid):
        """What a reviewer sees: each extracted term and the clause it came from."""
        return self.db.execute("""SELECT t.key, t.value, t.clause_number, c.text FROM terms t
            JOIN clauses c ON c.agreement_id=t.agreement_id AND c.number=t.clause_number
            WHERE t.agreement_id=? ORDER BY t.clause_number""", (aid,)).fetchall()

    def approve(self, aid, reviewer):
        self.db.execute("UPDATE agreements SET status='approved', approved_by=? WHERE id=?", (reviewer, aid))
        self.db.commit()

    def effective_clauses(self, merchant_id, as_of):
        """Approved docs in effect on as_of, oldest first; later clause numbers overwrite earlier ones."""
        docs = self.db.execute("""SELECT id FROM agreements WHERE merchant_id=? AND status='approved'
            AND effective_from<=? ORDER BY effective_from, id""", (merchant_id, as_of)).fetchall()
        merged = {}
        for (aid,) in docs:
            for n, title, text in self.db.execute("SELECT number,title,text FROM clauses WHERE agreement_id=?", (aid,)):
                merged[n] = {"id": n, "title": title, "text": text, "agreement_id": aid}
        return [merged[n] for n in sorted(merged)]

    def terms_as_of(self, merchant_id, as_of):
        out = {}
        for c in self.effective_clauses(merchant_id, as_of):
            for key, val, in self.db.execute("SELECT key,value FROM terms WHERE agreement_id=? AND clause_number=?",
                                             (c["agreement_id"], c["id"])):
                out[key] = {"value": val, "clause": c["id"], "agreement_id": c["agreement_id"]}
        return out

    def search(self, merchant_id, query, as_of, k=3):
        cl = self.effective_clauses(merchant_id, as_of)
        if not cl: return []
        docs = [f"{c['title']} {c['text']}" for c in cl]
        v = TfidfVectorizer(stop_words="english").fit(docs)
        s = cosine_similarity(v.transform([query]), v.transform(docs))[0]
        return [cl[i] for i in s.argsort()[::-1][:k] if s[i] > 0]

    def bot_for(self, merchant_id, as_of):
        """A SupportBot built from exactly what was in force on that date."""
        cl = self.effective_clauses(merchant_id, as_of)
        if not cl: raise LookupError(f"No approved agreement for {merchant_id} on {as_of}")
        return SupportBot("\n".join(f"{c['id']}. {c['title']}. {c['text']}" for c in cl))

if __name__ == "__main__":
    kb = AgreementKB()
    base, warn = kb.ingest("M1", AGREEMENT, "base", "2026-01-01", "m1_agreement.txt")
    print("Base warnings:", warn)
    print("Review sheet:")
    for key, val, cl, text in kb.review(base): print(f"  {key} = {val:g}  (clause {cl})")
    amend = ("1. Commission. Platform charges a commission of 15% on food subtotal.\n"
             "2. Preparation SLA. Partner shall have orders ready within 25 minutes of acceptance.")
    am, _ = kb.ingest("M1", amend, "amendment", "2026-07-01", "m1_amendment1.txt")

    print("\nBefore approval:", kb.terms_as_of("M1", "2026-03-01") or "nothing visible (drafts are hidden)")
    kb.approve(base, "legal.reviewer"); 
    print("Mar 2026:", {k: v["value"] for k, v in kb.terms_as_of("M1", "2026-03-01").items()})
    kb.approve(am, "legal.reviewer")
    t = kb.terms_as_of("M1", "2026-09-01")
    print("Sep 2026:", {k: v["value"] for k, v in t.items()})
    print("  SLA now from agreement", t["prep_sla_min"]["agreement_id"], "clause", t["prep_sla_min"]["clause"])

    print("\nSearch 'who pays for missing items':", [c["id"] for c in kb.search("M1", "who pays for missing items", "2026-09-01")])
    from support_bot import Order
    for when in ("2026-03-01", "2026-09-01"):
        d = kb.bot_for("M1", when).resolve("Order was late", Order(40, 33, 1))
        print(f"Late-prep dispute using {when} terms ->", d.outcome, f"${d.amount:.2f}")
