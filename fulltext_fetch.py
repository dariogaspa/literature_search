"""
fulltext_fetch.py
------------------
Recupero del testo completo di un paper. Strategia a cascata:

1. Se esiste un arXiv ID -> full text HTML via ar5iv.labs.arxiv.org
   (copre bene i paper astro-ph dal ~2007 in poi; per paper più vecchi
   o non convertiti correttamente, ar5iv può fallire).
2. Fallback: solo abstract (già presente nei metadati ADS).
3. Stub per full-text via l'export istituzionale ADS/editore: da
   completare con le credenziali del tuo istituto se vuoi coprire
   anche i paper senza controparte arXiv (es. vecchi paper pre-arXiv,
   o pubblicati solo su riviste chiuse).

Nota: l'angolo di vista (viewing angle) e i dettagli VLBI sono spesso
SOLO nel full text, non nell'abstract, quindi vale la pena investire
qui se questo campo è prioritario per voi.

Estrazione mirata dalle TABELLE: molti paper (soprattutto quelli a
campione statistico, es. cataloghi di centinaia di blazar) riportano
redshift/classificazione di una sorgente SOLO in una riga di tabella,
mai per nome nella prosa. Appiattire l'HTML in testo semplice (come
fa _extract_text_from_ar5iv_html) perde la corrispondenza riga/colonna,
quindi un LLM non può attribuire con sicurezza un valore numerico alla
sorgente giusta - e giustamente si rifiuta di indovinare, perdendo il
dato. extract_relevant_table_rows() cerca invece le tabelle HTML vere,
individua la riga che cita la sorgente, e la presenta al modello con la
sua intestazione di colonna intatta.
"""

import re
import time
import requests
from bs4 import BeautifulSoup

import vizier_table_lookup

AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"


def fetch_ar5iv_html(arxiv_id, retries=3, timeout=30):
    """Scarica l'HTML grezzo da ar5iv. Ritorna None se non disponibile."""
    if not arxiv_id:
        return None

    # ar5iv vuole l'id senza versione nella maggior parte dei casi
    clean_id = re.sub(r"v\d+$", "", arxiv_id)
    url = AR5IV_URL.format(arxiv_id=clean_id)

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "redshift-lit-pipeline/1.0"})
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"  [ar5iv] errore per {arxiv_id}: {e}, ritento tra {wait}s")
            time.sleep(wait)
    return None


def _extract_text_from_ar5iv_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # rimuove riferimenti bibliografici, formule renderizzate come immagini
    # opache, e altre sezioni poco utili per l'estrazione testuale
    for tag in soup.select("bibliography, .ltx_bibliography, .ltx_page_footer, script, style"):
        tag.decompose()

    article = soup.find("article") or soup.find("body") or soup
    text = article.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_for_match(s):
    return re.sub(r"\s+", "", (s or "").lower())


# Molte tabelle abbreviano gli identificatori omettendo il prefisso di
# catalogo (es. "3FGL J0312.7+3613" -> solo "J0312.7+3613" in tabella, per
# ragioni di spazio). Estraiamo il "token di coordinate" da ogni
# identificatore e lo usiamo come candidato di match aggiuntivo, oltre
# all'identificatore per intero.
_COORD_TOKEN_RE = re.compile(r"J\d{2,6}\.?\d*[+-]\d{2,6}\.?\d*", re.IGNORECASE)


def _coord_tokens(name):
    return _COORD_TOKEN_RE.findall(name or "")


def _build_match_candidates(names):
    """Costruisce l'insieme di stringhe normalizzate da cercare in una riga
    di tabella: ogni identificatore per intero, piu' il solo token di
    coordinate quando presente (per il caso in cui la tabella ometta il
    prefisso di catalogo)."""
    candidates = set()
    for n in names:
        if not n:
            continue
        norm = _normalize_for_match(n)
        if len(norm) >= 4:
            candidates.add(norm)
        for tok in _coord_tokens(n):
            norm_tok = _normalize_for_match(tok)
            # i token di sole coordinate sono corti e potenzialmente
            # ambigui (es. "J0312+36" potrebbe comparire per caso): si
            # richiede una lunghezza minima per ridurre i falsi positivi
            if len(norm_tok) >= 8:
                candidates.add(norm_tok)
    return candidates


_RA_KEY_RE = re.compile(r"[Jj](\d{4})")


def _extract_ra_keys(text):
    """Estrae le 'chiavi RA' (le 4 cifre di ore+minuti di ascensione retta
    subito dopo una 'J') da un testo. Deliberatamente grezzo e tollerante:
    non richiede che la dichinazione/segno siano ben formati subito dopo,
    perché il testo delle quote può arrivare "sporco" (segni meno unicode,
    spazi introdotti dalla conversione HTML/MathML di ar5iv)."""
    return set(_RA_KEY_RE.findall(text or ""))


def contains_foreign_source_reference(quote, identifiers=None, source_name=None):
    """
    Controlla se la citazione ('quote') restituita dal modello contiene un
    prefisso di coordinate RA (JHHMM...) che appartiene a un'ALTRA sorgente
    riconoscibile, diversa da quella richiesta.

    Serve a intercettare un fallimento diverso dalla classica allucinazione:
    il modello può citare testo REALMENTE presente nel chunk (superando
    quindi la verifica standard della quote in llm_extract.verify_quote),
    ma che si riferisce a una sorgente diversa - tipico quando un chunk di
    prosa non filtrato (a differenza dei blocchi tabella, già pre-filtrati
    sulla sorgente giusta) contiene i dati di più sorgenti mescolati
    insieme (es. una grande tabella appiattita in testo semplice).

    Si confronta solo il prefisso RA ore+minuti (4 cifre dopo la "J"), non
    l'intera stringa di coordinate: la declinazione/precisione può variare
    leggermente tra cataloghi per la STESSA sorgente (es. "J1838.0-5959"
    da un catalogo e "J183806.74-600032.1" da un altro sono la stessa
    posizione), mentre l'ora+minuti di ascensione retta resta stabile ed è
    già sufficiente a distinguere sorgenti in zone di cielo diverse (il
    caso osservato empiricamente: una quote corretta su "J1838...", una
    sbagliata dallo stesso paper su "J0002...", tutt'altra zona di cielo).
    """
    if not quote:
        return False

    names = ([source_name] if source_name else []) + list(identifiers or [])
    our_ra_keys = set()
    for n in names:
        our_ra_keys |= _extract_ra_keys(n)
    if not our_ra_keys:
        return False  # non riusciamo a stabilire le chiavi RA della sorgente: non blocchiamo per sicurezza

    quote_ra_keys = _extract_ra_keys(quote)
    foreign = quote_ra_keys - our_ra_keys
    return len(foreign) > 0


def quote_establishes_source_identity(quote, identifiers=None, source_name=None):
    """
    Controlla se la citazione ('quote') restituita dal modello nomina
    POSITIVAMENTE la sorgente richiesta (per nome/alias completo, o per
    il prefisso RA delle sue coordinate) - non solo che non ne nomini una
    sbagliata (vedi contains_foreign_source_reference, un controllo più
    debole: "non contiene un'altra sorgente" non implica "riguarda la
    sorgente giusta").

    Pensata per i chunk di PROSA non filtrati (chunks[:3] in pipeline.py):
    a differenza dei blocchi tabella - già pre-filtrati sulla sorgente
    giusta by construction, perché la riga è stata trovata cercando
    esplicitamente il suo nome - un chunk di prosa può contenere frasi
    tecniche (angolo di vista, fattore Doppler, redshift...) senza mai
    nominare esplicitamente a quale sorgente si riferiscono, specialmente
    in paper che confrontano più oggetti. Se la citazione non nomina la
    sorgente richiesta, non possiamo confermare l'attribuzione anche se
    il testo è verbatim presente nel chunk (caso osservato empiricamente:
    S4 0954+65, redshift preso da una citazione su "velocità
    superluminale... angolo di vista... fattore di Lorentz" che non cita
    mai il nome della sorgente né una sua coordinata).
    """
    if not quote:
        return False

    names = ([source_name] if source_name else []) + list(identifiers or [])
    candidates = _build_match_candidates(names)
    norm_quote = _normalize_for_match(quote)
    if any(c and c in norm_quote for c in candidates):
        return True

    our_ra_keys = set()
    for n in names:
        our_ra_keys |= _extract_ra_keys(n)
    quote_ra_keys = _extract_ra_keys(quote)
    return bool(quote_ra_keys & our_ra_keys)


def extract_relevant_table_rows(html, identifiers=None, source_name=None, max_tables=5):
    """
    Cerca nelle tabelle HTML del paper la/le riga/e che citano la sorgente
    (per nome o per uno dei suoi alias SIMBAD), e le ritorna come blocchi
    di testo compatti con l'intestazione delle colonne preservata:

        [Tabella 2 del paper]
        Intestazione colonne: Name | RA | Dec | z | Class
        Riga corrispondente alla sorgente: PKS 0017-307 | 00 19 42.6 | -30 31 20 | 0.677 | FSRQ

    Questo mantiene la corrispondenza riga/colonna che si perderebbe
    appiattendo la tabella in testo semplice, permettendo al modello di
    attribuire con sicurezza i valori numerici alla sorgente giusta.
    """
    names = [source_name] if source_name else []
    names += list(identifiers or [])
    norm_names = _build_match_candidates(names)
    if not norm_names:
        return []

    soup = BeautifulSoup(html, "html.parser")
    blocks = []

    for t_idx, table in enumerate(soup.find_all("table")):
        rows = table.find_all("tr")
        if not rows:
            continue

        header_cells = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
        header_line = " | ".join(header_cells) if header_cells else "(intestazione non trovata)"

        for row in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            if not cells:
                continue
            row_norm = _normalize_for_match(" ".join(cells))
            if any(nn in row_norm for nn in norm_names):
                row_line = " | ".join(cells)
                blocks.append(
                    f"[Tabella {t_idx + 1} del paper]\n"
                    f"Intestazione colonne: {header_line}\n"
                    f"Riga corrispondente alla sorgente: {row_line}"
                )
                if len(blocks) >= max_tables:
                    return blocks
    return blocks


def get_fulltext_from_ads_institutional(bibcode):
    """
    STUB da completare: molti istituti hanno accesso full-text via ADS
    "fulltext service" o via proxy dell'editore (A&A, ApJ, MNRAS...).
    Se la tua istituzione fornisce questo accesso, implementa qui la
    chiamata (tipicamente richiede EZproxy/Shibboleth o un export
    specifico) e ritorna il testo estratto, altrimenti None.
    """
    return None


def get_fulltext(paper, identifiers=None, source_name=None, max_chars=200_000):
    """
    Ritorna (testo, fonte, table_blocks) dove fonte è una delle stringhe:
    'arxiv_fulltext', 'ads_institutional', 'abstract_only'. table_blocks
    è una lista di blocchi di testo tabellare, da due fonti combinate:
    1. Tabelle HTML nel testo arXiv/ar5iv (extract_relevant_table_rows).
    2. Tabella dati integrale su VizieR, dedotta dal bibcode del paper
       (vizier_table_lookup): utile soprattutto per i paper a grande
       campione il cui preprint arXiv mostra solo un estratto di poche
       righe di esempio, rimandando alla tabella completa "in forma
       machine-readable" pubblicata a parte dalla rivista.
    Il testo è troncato a max_chars per contenere i costi di token
    in fase di estrazione LLM (i paper vengono comunque chunkati dopo).
    """
    table_blocks = []
    text, text_source = None, None

    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        html = fetch_ar5iv_html(arxiv_id)
        if html:
            text = _extract_text_from_ar5iv_html(html)
            table_blocks += extract_relevant_table_rows(html, identifiers, source_name)
            text_source = "arxiv_fulltext"

    try:
        table_blocks += vizier_table_lookup.find_source_row_in_paper_table(
            paper.get("bibcode"), identifiers=identifiers, source_name=source_name
        )
    except Exception as e:
        print(f"  [VizieR table lookup] errore per {paper.get('bibcode')}: {e}")

    if text:
        return text[:max_chars], text_source, table_blocks

    inst_text = get_fulltext_from_ads_institutional(paper.get("bibcode"))
    if inst_text:
        return inst_text[:max_chars], "ads_institutional", table_blocks

    return paper.get("abstract", "") or "", "abstract_only", table_blocks


def chunk_text(text, max_chars=24_000, overlap_chars=500):
    """
    Spezza un testo lungo in chunk con piccola sovrapposizione, per
    stare dentro il context window del modello LLM scelto senza
    tagliare frasi chiave (es. la sezione risultati/discussione dove
    di solito si trova il redshift o l'angolo di vista).
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap_chars
    return chunks
