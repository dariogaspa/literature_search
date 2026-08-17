"""
pipeline.py
-----------
Orchestrazione end-to-end:

  1. Legge la tabella con i risultati NED/SIMBAD/VizieR già prodotta
     (redshift_spectra_results.csv o simile) e seleziona le sorgenti
     che necessitano di ricerca in letteratura (Tier 3 / redshift
     mancante / classificazione incerta). Se il file non è trovato,
     processa tutte le sorgenti in new_5FGL_AGNs.csv.
  2. Per ciascuna sorgente: risolve gli identificatori via SIMBAD,
     cerca i paper rilevanti su ADS, recupera il full text (arXiv/ar5iv
     o abstract), chunka, ed esegue l'estrazione LLM per ogni chunk.
  3. Aggrega deterministicamente i risultati per sorgente.
  4. Salva due CSV (dettagliato per paper + aggregato per sorgente) con
     checkpoint incrementale per poter riprendere in caso di interruzione.

Uso:
    export ADS_TOKEN=...
    export ANTHROPIC_API_KEY=...      # se provider=anthropic
    export OPENAI_API_KEY=...         # se provider=openai
    python pipeline.py --provider anthropic --limit 50

Le variabili di configurazione principali sono in cima al file.
"""

import argparse
import os
import time
import pandas as pd

import ads_client
import fulltext_fetch
import llm_extract
import aggregate

# ====== CONFIGURAZIONE ======
INPUT_AGN_CSV = "new_5FGL_AGNs.csv"
EXISTING_RESULTS_CSV = "redshift_spectra_results.csv"  # output della pipeline precedente, se disponibile
DETAILED_OUTPUT_CSV = "literature_extraction_detailed.csv"
AGGREGATED_OUTPUT_CSV = "literature_results.csv"
CHECKPOINT_EVERY = 10
SLEEP_BETWEEN_SOURCES = 1.0
MAX_PAPERS_PER_SOURCE = 12
MIN_RELEVANCE_SCORE = 1


def select_sources_needing_literature(agn_csv, existing_results_csv):
    """
    Ritorna un DataFrame con le sole sorgenti da processare: quelle senza
    un redshift Tier 1/Tier 2 già affidabile, o con classificazione
    incerta/mancante. Se il file dei risultati precedenti non esiste,
    processa tutto l'elenco AGN.
    """
    agn_df = pd.read_csv(agn_csv)

    if not os.path.exists(existing_results_csv):
        print(f"[WARN] {existing_results_csv} non trovato: processo tutte le {len(agn_df)} sorgenti.")
        agn_df["_needs_literature"] = True
        return agn_df

    prev = pd.read_csv(existing_results_csv)
    merge_col_agn = "assoc" if "assoc" in agn_df.columns else agn_df.columns[0]
    if "assoc" in prev.columns:
        merge_col_prev = "assoc"
    elif "assoc_new" in prev.columns:
        merge_col_prev = "assoc_new"
    else:
        merge_col_prev = prev.columns[0]

    merged = agn_df.merge(
        prev[[merge_col_prev] + [c for c in prev.columns if c in
              ("z_best", "z_source", "classification")]],
        left_on=merge_col_agn, right_on=merge_col_prev, how="left"
    )

    tier3_sources = {"NED_nonS", "SIMBAD_DE", "2MRS", "6dFGS", "4LAC-DR3", None}

    def needs_lit(row):
        z_source = row.get("z_source")
        z_best = row.get("z_best")
        cls = row.get("classification")
        if pd.isna(z_best):
            return True
        if z_source in tier3_sources or pd.isna(z_source):
            return True
        if pd.isna(cls) or cls in ("AGN?", None, ""):
            return True
        return False

    merged["_needs_literature"] = merged.apply(needs_lit, axis=1)
    result = merged[merged["_needs_literature"]].copy()
    print(f"[INFO] {len(result)}/{len(agn_df)} sorgenti necessitano di ricerca in letteratura.")
    return result


def load_checkpoint(path):
    if os.path.exists(path):
        done = pd.read_csv(path)
        return done, set(done["source"].unique()) if "source" in done.columns else set()
    return None, set()


def process_source(source_name, provider, model):
    """Ritorna (detailed_rows, aggregated_row) per una singola sorgente."""
    identifiers = ads_client.get_identifiers(source_name)
    papers = ads_client.search_papers(source_name, identifiers=identifiers,
                                       max_results=MAX_PAPERS_PER_SOURCE * 2)
    papers = ads_client.rank_and_filter(papers, min_score=MIN_RELEVANCE_SCORE,
                                         max_papers=MAX_PAPERS_PER_SOURCE)

    detailed_rows = []
    extractions_for_agg = []
    n_errors_before = llm_extract.get_usage_totals()["n_errors"]

    for paper in papers:
        text, text_source, table_blocks = fulltext_fetch.get_fulltext(
            paper, identifiers=identifiers, source_name=source_name
        )
        if not text:
            continue

        chunks = fulltext_fetch.chunk_text(text)
        # I blocchi tabella sono compatti e mirati (una riga sola, gia'
        # filtrata per pertinenza alla sorgente), quindi li processiamo
        # sempre quando presenti, in aggiunta ai chunk di prosa - non in
        # sostituzione, perche' la stessa sorgente puo' avere sia una
        # tabella che una discussione testuale nello stesso paper.
        # Per contenere il numero di chiamate LLM, se il full text produce
        # molti chunk di prosa se ne processano solo i primi 3: risultati,
        # discussione e conclusioni tendono a comparire presto o in coda;
        # questo e' un compromesso costo/copertura regolabile.
        for chunk, is_table in [(c, True) for c in table_blocks] + [(c, False) for c in chunks[:3]]:
            # soglia di lunghezza minima della quote rilassata per i blocchi
            # tabella: sono gia' pre-filtrati sulla sorgente giusta (la riga
            # e' stata trovata cercando esplicitamente il suo nome), quindi
            # anche una citazione corta come "bll" o "IB" e' li' affidabile;
            # sulla prosa non filtrata si mantiene la soglia piena.
            min_len = 1 if is_table else 15
            result = llm_extract.extract_from_chunk(
                source_name, chunk, provider=provider, model=model, min_quote_length=min_len
            )
            if result is None:
                continue

            if not is_table and not fulltext_fetch.quote_establishes_source_identity(
                result.get("quote"), identifiers=identifiers, source_name=source_name
            ):
                # chunk di prosa non filtrato: la quote e' vera (ha superato
                # verify_quote) ma non nomina ne' la sorgente richiesta ne'
                # una sua coordinata riconoscibile - non possiamo confermare
                # che i valori estratti le appartengano davvero (potrebbero
                # riferirsi a un'altra sorgente discussa nello stesso paper,
                # es. un confronto multi-oggetto). I blocchi tabella sono
                # esenti da questo controllo: sono gia' garantiti pertinenti
                # per costruzione (la riga e' stata trovata cercando
                # esplicitamente il nome della sorgente).
                for k in ("redshift", "spectral_classification", "viewing_angle_deg"):
                    result[k] = None
                result["confidence"] = "source_identity_unconfirmed"
                result["quote_verified"] = False

            result.update({
                "source": source_name,
                "bibcode": paper.get("bibcode"),
                "year": paper.get("year"),
                "pub": paper.get("pub"),
                "text_source": text_source,
                # segnali per l'aggregazione pesata (vedi aggregate.py):
                # un'estrazione da blocco tabella e' spesso una riga
                # ereditata da un grande catalogo (poca garanzia di analisi
                # indipendente), un paper a campione statistico idem;
                # un'analisi dedicata in prosa su un paper non a campione
                # e' piu' probabile rifletta un giudizio autonomo.
                "is_table_derived": is_table,
                "is_bulk_sample_paper": ads_client.is_bulk_sample_paper(paper),
            })
            detailed_rows.append(result)
            extractions_for_agg.append(result)

    aggregated_row = aggregate.aggregate_source(source_name, extractions_for_agg)
    # Numero di chiamate LLM fallite (errori API, rate limit, credito
    # esaurito, ecc.) durante l'elaborazione di questa sorgente: permette
    # di distinguere "nessun dato trovato" (genuino) da "run compromesso
    # da errori API" nel CSV di output, invece di confonderli entrambi
    # come classification_final/redshift_final vuoti.
    aggregated_row["n_llm_errors"] = llm_extract.get_usage_totals()["n_errors"] - n_errors_before
    return detailed_rows, aggregated_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default=None, help="Override del modello di default del provider")
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di sorgenti (utile per test)")
    args = parser.parse_args()

    sources_df = select_sources_needing_literature(INPUT_AGN_CSV, EXISTING_RESULTS_CSV)
    if args.limit:
        sources_df = sources_df.head(args.limit)

    name_col = "assoc" if "assoc" in sources_df.columns else sources_df.columns[0]

    _, already_done = load_checkpoint(AGGREGATED_OUTPUT_CSV)
    if already_done:
        print(f"[INFO] Riprendo da checkpoint: {len(already_done)} sorgenti già processate.")

    # buffer che viene svuotato ad ogni checkpoint per evitare righe duplicate
    # quando si scrive in append mode
    detailed_buffer = []
    aggregated_buffer = []
    have_written_before = len(already_done) > 0
    n_since_checkpoint = 0

    for idx, row in sources_df.iterrows():
        source_name = row[name_col]
        if source_name in already_done:
            continue

        print(f"[{idx+1}/{len(sources_df)}] {source_name}")
        try:
            detailed_rows, aggregated_row = process_source(source_name, args.provider, args.model)
            detailed_buffer.extend(detailed_rows)
            aggregated_buffer.append(aggregated_row)
        except Exception as e:
            print(f"  [ERRORE] {source_name}: {e}")
            aggregated_buffer.append({"source": source_name, "error": str(e)})

        n_since_checkpoint += 1
        if n_since_checkpoint >= CHECKPOINT_EVERY:
            _save(detailed_buffer, aggregated_buffer, append=have_written_before)
            have_written_before = True
            detailed_buffer, aggregated_buffer = [], []
            n_since_checkpoint = 0
            _print_cost_so_far(args.provider, args.model)

        time.sleep(SLEEP_BETWEEN_SOURCES)

    if detailed_buffer or aggregated_buffer:
        _save(detailed_buffer, aggregated_buffer, append=have_written_before)
    _print_cost_so_far(args.provider, args.model)
    print("Completato.")


def _print_cost_so_far(provider, model):
    model_name = model or ("claude-sonnet-5" if provider == "anthropic" else "gpt-4.1")
    totals = llm_extract.get_usage_totals()
    cost = llm_extract.estimate_cost_usd(provider, model_name)
    cost_str = f"${cost:.3f}" if cost is not None else "n/d (modello non in tabella prezzi, vedi llm_extract.PRICING_PER_MTOK)"
    print(f"[COSTO] {totals['n_calls']} chiamate LLM, "
          f"{totals['input_tokens']} token input, {totals['output_tokens']} token output "
          f"-> stima costo: {cost_str}")
    if totals["n_errors"] > 0:
        print(f"[ATTENZIONE] {totals['n_errors']} chiamate LLM fallite con errore (rate limit, "
              f"credito esaurito, ecc.). Le sorgenti toccate da questi errori possono avere "
              f"classification_final/redshift_final vuoti per motivi NON genuini: controlla la "
              f"colonna n_llm_errors nell'output prima di interpretare i risultati come "
              f"'nessun dato trovato'.")


def _save(detailed_all, aggregated_all, append=False):
    if detailed_all:
        df = pd.DataFrame(detailed_all)
        df.to_csv(DETAILED_OUTPUT_CSV, mode="a" if append else "w",
                  header=not (append and os.path.exists(DETAILED_OUTPUT_CSV)), index=False)
    if aggregated_all:
        df = pd.DataFrame(aggregated_all)
        df.to_csv(AGGREGATED_OUTPUT_CSV, mode="a" if append else "w",
                  header=not (append and os.path.exists(AGGREGATED_OUTPUT_CSV)), index=False)


if __name__ == "__main__":
    main()
