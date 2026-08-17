"""
aggregate.py
------------
Aggregazione DETERMINISTICA (nessun LLM qui) delle estrazioni multiple
per una stessa sorgente. Le regole sono esplicite e documentate, cosa
che manca nell'approccio "a naso" descritto da Luigi: ogni valore finale
è tracciabile alla regola che lo ha prodotto e ai paper di origine.

Regole:
- redshift: tra le estrazioni con quote verificata, si preferisce
  redshift_type == 'spectroscopic'; a parità di tipo, si prende quella
  del paper più recente. Se due valori spettroscopici differiscono di
  più di 0.01 si segnala un conflitto (non si media/arrotonda).
- classificazione: REGOLA A DUE LIVELLI, non semplice conteggio di
  occorrenze. Un'estrazione è "dedicata" se NON viene da una riga di
  tabella (is_table_derived=False) E NON viene da un paper esplicitamente
  a campione statistico (is_bulk_sample_paper=False): cioè un'analisi in
  prosa su un paper mirato alla singola sorgente, la firma di un giudizio
  originale piuttosto che la ripetizione di un'etichetta ereditata da un
  catalogo. Se esiste ALMENO UN'estrazione dedicata, la classificazione
  finale si decide SOLO tra quelle (maggioranza, tie-break sul paper più
  recente), ignorando le menzioni ereditate anche se numericamente
  maggioritarie. Solo se non esiste nessuna estrazione dedicata si
  ripiega sul voto tra tutte le estrazioni disponibili. Motivazione
  osservata empiricamente: caso TXS 1206+549, dove 13 menzioni "FSRQ"
  ereditate da tabelle/cataloghi battevano numericamente l'unico paper
  con spettroscopia dedicata che classificava correttamente la sorgente
  come NLS1 - un semplice conteggio (o persino un voto pesato con
  penalità moderate) non basta a risolvere correttamente questo caso,
  perché le menzioni ereditate non sono evidenza indipendente: sono
  quasi certamente tutte la stessa etichetta ricopiata dalla stessa
  fonte originale, non 13 verifiche separate. Si riporta comunque lo
  storico completo (incluse le menzioni ereditate) se la classificazione
  cambia nel tempo.
- viewing angle: NON si fonde tra metodi diversi; si riportano tutte le
  stime verificate con il rispettivo metodo e paper di origine.
"""

from collections import Counter

CLASS_NORMALIZATION = {
    "bl lac": "BL Lac", "bll": "BL Lac", "bl lac object": "BL Lac",
    "fsrq": "FSRQ", "flat-spectrum radio quasar": "FSRQ",
    "flat spectrum radio quasar": "FSRQ",
    "radio galaxy": "Radio Galaxy", "narrow-line radio galaxy": "Radio Galaxy",
    "seyfert": "Seyfert", "seyfert 1": "Seyfert", "seyfert 2": "Seyfert",
    "blazar": "Blazar (unclassified)",
}


def normalize_classification(raw):
    if not raw:
        return None
    key = raw.strip().lower()
    return CLASS_NORMALIZATION.get(key, raw.strip())


def _is_dedicated(e):
    """True se l'estrazione ha la firma di un'analisi dedicata (prosa, non
    tabella; paper non esplicitamente a campione statistico) piuttosto che
    di una menzione ereditata da un catalogo. Campi assenti (dati più
    vecchi) sono trattati come False per entrambi i segnali, cioè
    l'estrazione è considerata dedicata di default - nessuna
    penalizzazione retroattiva su dati per cui non abbiamo il segnale."""
    return not e.get("is_table_derived", False) and not e.get("is_bulk_sample_paper", False)


def aggregate_source(source_name, extractions):
    """
    extractions: lista di dict, ciascuno il risultato di extract_from_chunk
    arricchito con 'year' e 'bibcode' del paper di origine, e (quando
    disponibili) 'is_table_derived'/'is_bulk_sample_paper' per la regola
    a due livelli sulla classificazione (vedi _is_dedicated).
    """
    verified = [e for e in extractions if e.get("quote_verified")]

    out = {
        "source": source_name,
        "n_papers_reviewed": len({e.get("bibcode") for e in extractions if e.get("bibcode")}),
        "n_extractions_verified": len(verified),
        "redshift_final": None,
        "redshift_source_bibcode": None,
        "redshift_conflict": False,
        "classification_final": None,
        "classification_agreement": None,
        "classification_history": None,
        "viewing_angle_estimates": None,
    }

    # --- redshift ---
    z_candidates = [e for e in verified if e.get("redshift") is not None]
    if z_candidates:
        spec = [e for e in z_candidates if e.get("redshift_type") == "spectroscopic"]
        pool = spec if spec else z_candidates
        pool_sorted = sorted(pool, key=lambda e: e.get("year") or 0, reverse=True)
        best = pool_sorted[0]
        out["redshift_final"] = best.get("redshift")
        out["redshift_source_bibcode"] = best.get("bibcode")

        values = {round(e["redshift"], 3) for e in pool}
        if len(values) > 1 and max(values) - min(values) > 0.01:
            out["redshift_conflict"] = True

    # --- classificazione (regola a due livelli, vedi _is_dedicated) ---
    cls_candidates = [
        (normalize_classification(e.get("spectral_classification")), e.get("year") or 0,
         e.get("bibcode"), _is_dedicated(e))
        for e in verified if e.get("spectral_classification")
    ]
    if cls_candidates:
        dedicated = [c for c in cls_candidates if c[3]]
        pool = dedicated if dedicated else cls_candidates
        tier_label = "analisi dedicate" if dedicated else "tutte le estrazioni, nessuna analisi dedicata trovata"

        counts = Counter(c[0] for c in pool)
        top_count = max(counts.values())
        top_classes = [c for c, n in counts.items() if n == top_count]
        if len(top_classes) == 1:
            chosen = top_classes[0]
        else:
            # pareggio: vince la classificazione del paper più recente NEL POOL usato
            pool_sorted = sorted(pool, key=lambda t: t[1], reverse=True)
            chosen = next(c for c, *_ in pool_sorted if c in top_classes)

        out["classification_final"] = chosen
        out["classification_agreement"] = (
            f"{counts[chosen]}/{len(pool)} ({tier_label}; {len(cls_candidates)} estrazioni totali)"
        )

        history_entries = sorted({(c, y, b) for c, y, b, _ in cls_candidates}, key=lambda t: t[1])
        if len({c for c, _, _ in history_entries}) > 1:
            out["classification_history"] = "; ".join(f"{y}:{c}" for c, y, _ in history_entries)

    # --- viewing angle ---
    va_candidates = [
        e for e in verified
        if e.get("viewing_angle_deg") is not None
    ]
    if va_candidates:
        out["viewing_angle_estimates"] = "; ".join(
            f"{e['viewing_angle_deg']} deg ({e.get('viewing_angle_method') or 'unknown method'}, "
            f"{e.get('bibcode') or '?'})"
            for e in va_candidates
        )

    return out
