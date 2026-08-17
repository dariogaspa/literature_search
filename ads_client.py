"""
ads_client.py
-------------
Wrapper per le query a NASA ADS (Search API), con risoluzione degli
identificatori alternativi via SIMBAD (un blazar ha spesso 5-6 nomi
diversi tra cataloghi radio/ottici/X, e cercare solo il nome Fermi
fa perdere gran parte della letteratura pre-Fermi).

Serve un token ADS: https://ui.adsabs.harvard.edu/user/settings/token
Da passare via variabile d'ambiente ADS_TOKEN.
"""

import os
import re
import time
import requests
from astroquery.simbad import Simbad

ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"

# Parole chiave usate per giudicare la rilevanza di un paper rispetto
# al task (redshift, classificazione, angolo di vista). Non sono un
# filtro rigido: servono per ordinare/pesare i risultati.
RELEVANCE_KEYWORDS = [
    "redshift", "spectrum", "spectroscop", "classification",
    "blazar", "bl lac", "fsrq", "viewing angle", "jet", "vlbi",
    "superluminal", "beaming", "doppler factor", "sed",
    # aggiunti dopo aver scoperto (vedi TXS 1206+549 / B2 1100+30B) che la
    # lista originale, centrata sui blazar, dava punteggio troppo basso a
    # paper di classificazione Seyfert/NLS1/CSS altrettanto pertinenti
    "seyfert", "nls1", "narrow-line", "narrow line", "compact steep spectrum",
    "changing-look", "changing look",
]

# Titoli con queste espressioni sono tipicamente studi a campione su
# centinaia/migliaia di sorgenti: la classificazione/redshift della
# singola sorgente cercata di solito finisce in una tabella (dati
# supplementari), non discussa per nome nella prosa dell'articolo. Non
# li escludiamo (potrebbero comunque contenere l'informazione), ma li
# penalizziamo nel ranking rispetto a studi mirati sulla singola sorgente.
BULK_SAMPLE_PENALTY_KEYWORDS = [
    "sample of", "catalog of", "catalogue of", "population of",
    "machine learning", "monitoring program", "survey of", "large sample",
]


def get_ads_token():
    token = os.environ.get("ADS_TOKEN")
    if not token:
        raise RuntimeError(
            "Variabile d'ambiente ADS_TOKEN non impostata. "
            "Genera un token su https://ui.adsabs.harvard.edu/user/settings/token"
        )
    return token


def get_identifiers(source_name, extra_names=None, retries=3):
    """
    Recupera gli identificatori alternativi di una sorgente da SIMBAD
    (es. per un blazar: nome Fermi, nome radio, nome ottico, ecc.).
    Ritorna una lista di stringhe (incluso il nome originale).
    """
    names = set()
    if source_name:
        names.add(source_name)
    if extra_names:
        names.update(extra_names)

    for attempt in range(retries):
        try:
            result = Simbad.query_objectids(source_name)
            if result is not None:
                col = "ID" if "ID" in result.colnames else result.colnames[0]
                for row in result[col]:
                    val = row.decode() if isinstance(row, bytes) else str(row)
                    names.add(val.strip())
            break
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [SIMBAD ids] errore per {source_name}: {e}, ritento tra {wait}s")
            time.sleep(wait)

    return list(names)


# Dentro una query di frase tra virgolette (es. object:"PKS 0017-307"), la
# stragrande maggioranza dei caratteri "speciali" Lucene (- + & | ecc.) NON
# va escapata: perde già il significato speciale essendo dentro le
# virgolette, e aggiungere un backslash produce una sequenza di escape
# non valida che ADS rifiuta con 400 (es. "PKS 0017\-307" è sbagliato).
# Vanno escapati solo i caratteri che romperebbero letteralmente la frase:
# il backslash stesso e il doppio apice.
_LUCENE_SPECIAL_CHARS = '"\\'


def _escape_lucene(text):
    escaped = []
    for ch in text:
        if ch in _LUCENE_SPECIAL_CHARS:
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)


# Massimo numero di identificatori da includere nella query OR: SIMBAD può
# restituirne 20-30 per un blazar ben studiato, ma una query con troppe
# clausole OR (o troppo lunga) viene rifiutata da ADS con un 400. Si dà
# priorità agli identificatori più corti/semplici, che tendono a
# corrispondere ai nomi da catalogo più standard (PKS, TXS, QSO, ecc.)
# piuttosto che alle designazioni bibliografiche tra parentesi quadre.
MAX_IDENTIFIERS_IN_QUERY = 8


# Cataloghi/survey dedicati ad AGN e blazar: i loro identificatori
# compaiono tipicamente nel testo dei paper (spesso il paper stesso è
# quello che ha introdotto il catalogo, es. CGRaBS = Healey et al. 2008).
# Vengono quindi privilegiati rispetto a designazioni di posizione da
# survey generiche multi-banda (WISEA, 1eRASS, GALEXASC, SUMSS, NVSS,
# AT20G, TGSSADR...), che sono cross-match automatici quasi mai citati
# per nome nel testo di un paper.
_PREFERRED_CATALOG_PREFIXES = (
    "PKS", "TXS", "CGRaBS", "CRATES", "BZQ", "BZB", "BZG", "BZU",
    "MILLIQUAS", "MQS", "S5", "OJ", "PG", "B2", "B3", "4FGL", "3FGL",
    "QSO", "IERS", "ICRF", "IVS", "VCS", "1Jy", "PMN",
)


def _strip_simbad_bib_prefix(identifier):
    """
    SIMBAD annota alcuni identificatori con un prefisso tra parentesi
    quadre tipo "[MGL2009] BZQ J0019-3031": il prefisso indica solo QUALE
    paper ha introdotto quel nome (qui Massaro/Giommi/... 2009), non fa
    parte del nome usato nel testo del paper stesso. Cercare la stringa
    con le parentesi non trova quindi nulla: si tiene solo la parte dopo
    il prefisso, che è il nome effettivamente usato in letteratura.
    """
    m = re.match(r"^\[[^\]]+\]\s*(.+)$", identifier)
    return m.group(1) if m else identifier


def _catalog_priority(identifier):
    """0 = catalogo dedicato AGN/blazar (alta probabilità di comparire nei
    paper), 1 = tutto il resto (survey generiche, designazioni di posizione)."""
    upper = identifier.upper()
    for prefix in _PREFERRED_CATALOG_PREFIXES:
        if upper.startswith(prefix.upper()):
            return 0
    return 1


def _select_best_identifiers(identifiers, max_n=MAX_IDENTIFIERS_IN_QUERY):
    cleaned = [_strip_simbad_bib_prefix(i) for i in identifiers if i and i.strip()]
    # rimuove duplicati che possono emergere dopo lo strip (es. due
    # identificatori bibliografici diversi che puntano allo stesso nome)
    seen = set()
    deduped = []
    for i in cleaned:
        if i not in seen:
            seen.add(i)
            deduped.append(i)

    ordered = sorted(deduped, key=lambda i: (_catalog_priority(i), len(i)))
    return ordered[:max_n]


def _build_object_query(identifiers):
    """
    Costruisce la query ADS come OR di frasi tra virgolette sui vari
    identificatori: "id1" OR "id2" OR ...

    NOTA: il campo object: (usato dalla barra di ricerca del sito ADS)
    NON è un campo Solr valido per l'API diretta /v1/search/query — lì
    dietro c'è un microservizio SIMBAD/NED interno alla webapp che non è
    esposto pubblicamente. Chiamarlo con object: dà "undefined field
    object" (400). Una ricerca a frase semplice (senza prefisso di campo)
    cerca invece in titolo/abstract/keyword ed è supportata nativamente
    dall'API pubblica: meno precisa del tagging SIMBAD/NED, ma cattura la
    grande maggioranza dei paper che citano la sorgente per nome.
    """
    selected = _select_best_identifiers(identifiers)
    parts = [f'"{_escape_lucene(ident)}"' for ident in selected]
    return " OR ".join(parts)


def _run_query(q, max_results, retries, timeout):
    """Esegue una singola query ADS con retry su errori di rete/rate-limit.
    Ritorna (docs, bad_request) dove bad_request=True se ADS risponde 400
    (in tal caso non ha senso ritentare la STESSA query: va accorciata a monte)."""
    token = get_ads_token()
    params = {
        "q": q,
        "fl": "bibcode,title,abstract,year,identifier,doctype,pub,property",
        "rows": max_results,
        "sort": "date desc",
    }

    for attempt in range(retries):
        try:
            r = requests.get(
                ADS_API_URL,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=timeout,
            )
            if r.status_code == 400:
                print(f"  [ADS] 400 Bad Request. URL: {r.url}")
                print(f"  [ADS] corpo risposta: {r.text[:1000]}")
                return [], True
            if r.status_code == 401:
                print("  [ADS] 401 Unauthorized: controlla che ADS_TOKEN sia corretto e attivo.")
                return [], False
            if r.status_code == 429:
                wait = min(60, 2 ** attempt * 5)
                print(f"  [ADS] rate limited, aspetto {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            docs = r.json().get("response", {}).get("docs", [])
            return [_normalize_doc(d) for d in docs], False
        except requests.exceptions.RequestException as e:
            wait = min(60, 2 ** attempt)
            print(f"  [ADS] errore ({e}), ritento tra {wait}s...")
            time.sleep(wait)

    print(f"  [ADS] fallito dopo {retries} tentativi")
    return [], False


def get_bibcodes_from_simbad(source_name, retries=3, timeout=30):
    """
    Recupera i bibcode dei paper che SIMBAD ha collegato a questa sorgente
    (bibliografia curata: qualcuno ha letto il paper e taggato l'oggetto).
    È la STESSA fonte dati che il modificatore object: usa sul sito ADS,
    ma qui la raggiungiamo direttamente via TAP/ADQL, così non dipendiamo
    da un campo che l'API diretta di ADS non espone.

    Ritorna una lista di dict {bibcode, year} ordinata per anno decrescente,
    oppure [] se l'oggetto non è in SIMBAD o non ha bibliografia associata.
    """
    safe_name = source_name.replace("'", "''")
    query = f"""
    SELECT ref.bibcode, ref."year" AS pubyear
    FROM ident
    JOIN has_ref ON has_ref.oidref = ident.oidref
    JOIN ref ON ref.oidbib = has_ref.oidbibref
    WHERE ident.id = '{safe_name}'
    ORDER BY pubyear DESC
    """

    for attempt in range(retries):
        try:
            table = Simbad.query_tap(query)
            if table is None or len(table) == 0:
                return []
            entries = []
            for row in table:
                try:
                    year = int(row["pubyear"])
                except (TypeError, ValueError):
                    year = None
                entries.append({"bibcode": str(row["bibcode"]), "year": year})
            return entries
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [SIMBAD TAP] errore bibliografia per {source_name}: {e}, ritento tra {wait}s")
            time.sleep(wait)

    print(f"  [SIMBAD TAP] fallito dopo {retries} tentativi per {source_name}")
    return []


def get_papers_by_bibcodes(bibcodes, batch_size=15, retries=5, timeout=60):
    """
    Recupera titolo/abstract/anno/arxiv_id per una lista di bibcode usando
    il campo bibcode: (un campo Solr reale, a differenza di object:).
    Query in batch per stare dentro ai limiti di lunghezza dell'URL.
    """
    all_docs = []
    for i in range(0, len(bibcodes), batch_size):
        batch = bibcodes[i:i + batch_size]
        q = " OR ".join(f'bibcode:"{_escape_lucene(b)}"' for b in batch)
        docs, bad_request = _run_query(q, max_results=batch_size, retries=retries, timeout=timeout)
        if bad_request:
            print(f"  [ADS] 400 anche sulla query bibcode: (batch di {len(batch)}), salto il batch.")
            continue
        all_docs.extend(docs)
    return all_docs


def search_papers(source_name, identifiers=None, max_results=25,
                   year_min=None, retries=5, timeout=60, max_bibcodes=100):
    """
    Cerca su ADS tutti i paper associati a una sorgente.

    Strategia (in ordine):
    1. Bibliografia SIMBAD via TAP (fonte curata, la più affidabile) -> se
       trova bibcode, recupera i metadati da ADS via bibcode: (batch).
    2. Fallback: ricerca a frase sui nomi/identificatori (meno precisa,
       cerca solo in titolo/abstract/keyword, utile se l'oggetto non è
       in SIMBAD o non ha bibliografia associata).
    """
    identifiers = identifiers or [source_name]

    bib_entries = get_bibcodes_from_simbad(source_name, retries=retries, timeout=timeout)
    if bib_entries:
        bibcodes = [e["bibcode"] for e in bib_entries[:max_bibcodes]]
        docs = get_papers_by_bibcodes(bibcodes, retries=retries, timeout=timeout)
        if docs:
            return docs
        print(f"  [ADS] bibcode SIMBAD trovati ({len(bibcodes)}) ma nessun metadato recuperato da ADS, provo il fallback testuale.")

    # fallback: ricerca a frase progressivamente più ristretta
    fallback_counts = [MAX_IDENTIFIERS_IN_QUERY, 4, 2, 1]
    for n in fallback_counts:
        subset = _select_best_identifiers(identifiers, max_n=n) or [source_name]
        q = _build_object_query(subset)
        if not q:
            continue
        if year_min:
            q = f"({q}) AND year:[{year_min} TO 9999]"

        docs, bad_request = _run_query(q, max_results, retries, timeout)
        if bad_request:
            print(f"  [ADS] riprovo con meno identificatori (n={n})...")
            continue
        return docs

    print(f"  [ADS] tutte le strategie hanno fallito per {source_name}")
    return []


def _normalize_doc(doc):
    arxiv_id = None
    for ident in doc.get("identifier", []) or []:
        if ident.lower().startswith("arxiv:"):
            arxiv_id = ident.split(":", 1)[1]
            break
        # a volte l'identifier è tipo "2020arXiv200101234S" -> non utile;
        # il formato "arXiv:XXXX.XXXXX" è quello standard restituito da ADS.

    title = doc.get("title", [""])
    title = title[0] if isinstance(title, list) and title else (title or "")
    abstract = doc.get("abstract", "") or ""

    return {
        "bibcode": doc.get("bibcode"),
        "title": title,
        "abstract": abstract,
        "year": doc.get("year"),
        "pub": doc.get("pub"),
        "doctype": doc.get("doctype"),
        "arxiv_id": arxiv_id,
    }


def any_identifier_in_text(identifiers, text, source_name=None):
    """
    Controlla se il nome della sorgente o uno dei suoi alias SIMBAD compare
    letteralmente nel testo (case-insensitive, tollerante a spazi/assenza
    di spazi tipo "PKS 0017-307" vs "PKS0017-307"). Usato per evitare di
    spendere una chiamata LLM su chunk che quasi certamente non discutono
    la sorgente per nome (es. tabelle di grandi campioni dove il testo in
    prosa non la menziona mai).
    """
    if not text:
        return False
    norm_text = re.sub(r"\s+", "", text.lower())

    names = list(identifiers or [])
    if source_name:
        names.append(source_name)

    for name in names:
        clean = _strip_simbad_bib_prefix(name)
        norm_name = re.sub(r"\s+", "", clean.lower())
        if len(norm_name) >= 4 and norm_name in norm_text:
            return True
    return False


def relevance_score(paper):
    """Punteggio di rilevanza di BASE (solo parole chiave in titolo+abstract).
    Usato per la soglia di esclusione min_score in rank_and_filter: non
    include la penalita' per i paper a campione statistico, altrimenti
    rischierebbe di escludere anche gli unici paper disponibili per una
    sorgente invece di limitarsi a deprioritizzarli nell'ordinamento."""
    text = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
    return sum(1 for kw in RELEVANCE_KEYWORDS if kw in text)


def is_bulk_sample_paper(paper):
    """True se il titolo del paper contiene espressioni tipiche degli
    studi a campione statistico (vedi BULK_SAMPLE_PENALTY_KEYWORDS).
    Esposta come funzione pubblica perche' riusata sia per il ranking
    (_sort_score) sia come segnale per l'aggregazione pesata delle
    classificazioni in aggregate.py."""
    title = (paper.get("title") or "").lower()
    return any(kw in title for kw in BULK_SAMPLE_PENALTY_KEYWORDS)


def _sort_score(paper):
    """Punteggio usato SOLO per l'ordinamento (non per l'esclusione): al
    punteggio di base sottrae la penalita' per i paper a campione
    statistico, cosi' un paper mirato viene scelto per primo quando ce ne
    sono altri disponibili, ma un paper a campione non viene mai escluso
    se e' l'unico che parla della sorgente."""
    score = relevance_score(paper)
    if is_bulk_sample_paper(paper):
        score -= 2
    return score


def rank_and_filter(papers, min_score=1, max_papers=15):
    """Filtra i paper per rilevanza di base (esclusione), poi li ordina
    usando il punteggio penalizzato (i paper a campione statistico vengono
    messi in coda rispetto a studi mirati sulla singola sorgente, ma non
    esclusi se sono gli unici disponibili)."""
    scored = [(relevance_score(p), p) for p in papers]
    scored = [(s, p) for s, p in scored if s >= min_score]
    scored.sort(key=lambda sp: (_sort_score(sp[1]), sp[1].get("year") or 0), reverse=True)
    return [p for _, p in scored[:max_papers]]
