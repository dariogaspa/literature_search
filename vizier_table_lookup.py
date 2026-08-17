"""
vizier_table_lookup.py
------------------------
Molti paper "a campione" (centinaia di sorgenti) su ApJ/ApJS/MNRAS/A&A
pubblicano la tabella dati COMPLETA solo come catalogo VizieR a se'
stante, non nel testo del preprint arXiv - che spesso mostra solo un
estratto di poche righe di esempio, con una nota tipo "the table is
available in its entirety in machine-readable form" (visto empiricamente
su Fan et al. 2016, ApJS 226, 20, durante lo sviluppo di questa pipeline).

Questo modulo deriva l'identificativo VizieR piu' probabile direttamente
dal bibcode ADS del paper - la convenzione di naming di VizieR
"J/<rivista>/<volume>/<pagina>" coincide quasi sempre con gli estremi
bibliografici del bibcode stesso - e cerca la riga della sorgente in
quella tabella, con lo stesso approccio di matching (nome completo o
solo token di coordinate) usato per le tabelle HTML di fulltext_fetch.py.
"""

import re
from astroquery.vizier import Vizier

# Mappa dal codice rivista usato nei bibcode ADS al codice usato nei
# path dei cataloghi VizieR. Non e' garantito 1:1 in tutti i casi (VizieR
# ha le sue convenzioni storiche per riviste minori), ma copre le
# riviste piu' comuni in astrofisica extragalattica/AGN.
_JOURNAL_ADS_TO_VIZIER = {
    "ApJ": "ApJ", "ApJS": "ApJS", "ApJL": "ApJ",
    "MNRAS": "MNRAS", "A&A": "A+A", "AJ": "AJ", "PASP": "PASP",
    "Natur": "Nat", "Sci": "Sci", "RAA": "RAA", "PASJ": "PASJ",
}

_COORD_TOKEN_RE = re.compile(r"J\d{2,6}\.?\d*[+-]\d{2,6}\.?\d*", re.IGNORECASE)


def bibcode_to_vizier_catalog_id(bibcode):
    """
    Deriva l'identificativo VizieR J/<rivista>/<volume>/<pagina> dal
    bibcode ADS standard a 19 caratteri (YYYY JJJJJ VVVV M PPPP A).
    Ritorna None se il bibcode non e' nel formato standard o la rivista
    non e' nella tabella di conversione.
    """
    if not bibcode or len(bibcode) != 19:
        return None
    journal_raw = bibcode[4:9].replace(".", "")
    volume_raw = bibcode[9:13].replace(".", "")
    page_raw = bibcode[14:18].replace(".", "")

    journal = _JOURNAL_ADS_TO_VIZIER.get(journal_raw)
    if not journal or not volume_raw or not page_raw:
        return None
    return f"J/{journal}/{volume_raw}/{page_raw}"


def _normalize(s):
    return re.sub(r"\s+", "", str(s or "").lower())


def _match_candidates(identifiers, source_name=None):
    """Stessa logica di fulltext_fetch._build_match_candidates: nome
    completo o solo token di coordinate (per i casi in cui la tabella
    ometta il prefisso di catalogo)."""
    names = [source_name] if source_name else []
    names += list(identifiers or [])
    candidates = set()
    for n in names:
        if not n:
            continue
        norm = _normalize(n)
        if len(norm) >= 4:
            candidates.add(norm)
        for tok in _COORD_TOKEN_RE.findall(n):
            norm_tok = _normalize(tok)
            if len(norm_tok) >= 8:
                candidates.add(norm_tok)
    return candidates


def find_source_row_in_paper_table(bibcode, identifiers=None, source_name=None,
                                    max_tables=3, timeout=60):
    """
    Cerca la riga della sorgente nella tabella dati integrale depositata
    su VizieR per questo paper (catalogo dedotto dal bibcode). Ritorna
    una lista di blocchi di testo (stesso formato di
    fulltext_fetch.extract_relevant_table_rows: intestazione colonne +
    riga pertinente), vuota se il catalogo non esiste su VizieR, non
    contiene la sorgente, o il bibcode non e' nel formato atteso.
    """
    catalog_id = bibcode_to_vizier_catalog_id(bibcode)
    if not catalog_id:
        return []

    candidates = _match_candidates(identifiers, source_name)
    if not candidates:
        return []

    Vizier.ROW_LIMIT = -1  # serve poter cercare in tutta la tabella, non solo le prime righe
    try:
        tables = Vizier.get_catalogs(catalog_id)
    except Exception as e:
        print(f"  [VizieR] nessun catalogo trovato per {catalog_id} (bibcode {bibcode}): {e}")
        return []

    if not tables:
        return []

    blocks = []
    for t_idx, table in enumerate(tables):
        if len(blocks) >= max_tables:
            break
        colnames = table.colnames
        for row in table:
            row_values = [str(row[c]) for c in colnames]
            row_norm = _normalize(" ".join(row_values))
            if any(c in row_norm for c in candidates):
                header_line = " | ".join(colnames)
                row_line = " | ".join(row_values)
                blocks.append(
                    f"[Tabella VizieR {catalog_id}, dataset {t_idx + 1} - "
                    f"tabella integrale del paper, non presente nel testo arXiv]\n"
                    f"Intestazione colonne: {header_line}\n"
                    f"Riga corrispondente alla sorgente: {row_line}"
                )
                break  # una riga per dataset e' sufficiente
    return blocks
