"""
diagnose.py
-----------
Diagnostica passo-passo su UNA sola sorgente nota, per capire dove si
interrompe la catena quando il risultato finale è "0 paper trovati".

Uso:
    export ADS_TOKEN=...
    python diagnose.py
"""

import ads_client
import fulltext_fetch

import sys

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "PKS 0017-307"

print(f"=== Diagnostica per: {SOURCE} ===\n")

print("1) Bibliografia SIMBAD via TAP (fonte primaria, curata)...")
bib_entries = ads_client.get_bibcodes_from_simbad(SOURCE, retries=2, timeout=30)
print(f"   n_bibcodes trovati: {len(bib_entries)}")
if bib_entries:
    for e in bib_entries[:8]:
        print(f"   - {e['bibcode']} ({e['year']})")
print()

print("1b) Query ADS diretta (solo il nome, ricerca a frase di fallback)...")
docs, bad_request = ads_client._run_query(
    ads_client._build_object_query([SOURCE]),
    max_results=25, retries=2, timeout=30,
)
print(f"   bad_request={bad_request}, n_docs={len(docs)}")
if docs:
    for d in docs[:5]:
        print(f"   - {d['bibcode']} ({d['year']}): {d['title'][:80]}")
print()

print("2) get_identifiers via SIMBAD...")
identifiers = ads_client.get_identifiers(SOURCE)
print(f"   n_identifiers={len(identifiers)}")
print(f"   esempio: {identifiers[:10]}")
print()

print("3) search_papers (con fallback automatico)...")
papers = ads_client.search_papers(SOURCE, identifiers=identifiers, max_results=25)
print(f"   n_papers grezzi: {len(papers)}")
print()

print("4) rank_and_filter (min_score=1)...")
filtered = ads_client.rank_and_filter(papers, min_score=1, max_papers=12)
print(f"   n_papers dopo filtro rilevanza: {len(filtered)}")
if papers and not filtered:
    print("   [!] Il filtro di rilevanza sta scartando TUTTI i paper trovati.")
    print("   Punteggi dei primi 5 paper grezzi:")
    for p in papers[:5]:
        print(f"     score={ads_client.relevance_score(p)} | {p['title'][:80]}")
print()

if filtered:
    print("5) Fetch full text del primo paper filtrato...")
    paper = filtered[0]
    text, text_source, table_blocks = fulltext_fetch.get_fulltext(
        paper, identifiers=identifiers, source_name=SOURCE
    )
    print(f"   blocchi tabella trovati (righe pertinenti alla sorgente): {len(table_blocks)}")
    for b in table_blocks:
        print(f"   {b}")
    print(f"   bibcode={paper.get('bibcode')}, arxiv_id={paper.get('arxiv_id')}")
    print(f"   text_source={text_source}, lunghezza testo={len(text) if text else 0}")
    print(f"   primi 300 caratteri: {(text or '')[:300]}")
    print()

    print("6) Estrazione LLM sul testo recuperato...")
    import llm_extract
    if table_blocks:
        chunk = table_blocks[0]
        print("   (testo il blocco TABELLA trovato, come fa realmente pipeline.py)")
    else:
        chunk = fulltext_fetch.chunk_text(text)[0] if text else ""
        print("   (nessun blocco tabella: testo il primo chunk di prosa)")
    result = llm_extract.extract_from_chunk(SOURCE, chunk, provider="anthropic")
    print(f"   risultato: {result}")
    print(f"   usage totali finora: {llm_extract.get_usage_totals()}")

print("\n=== Fine diagnostica ===")
