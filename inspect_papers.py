"""
inspect_papers.py
------------------
Rilancia la pipeline SOLO sulle sorgenti indicate (invece di un intero
batch/campione) e stampa il dettaglio completo per-paper: bibcode, anno,
classificazione estratta, quote a supporto. Utile per ispezionare a mano
un disaccordo con Foschini prima di decidere se e come affinare la
regola di aggregazione, senza il costo di rilanciare l'intero campione.

Uso:
    export ADS_TOKEN=...
    export ANTHROPIC_API_KEY=...
    python inspect_papers.py "SUMSS J183806-600033" "RX J0814.4+2941"
"""

import sys
import pipeline as lit_pipeline
import llm_extract


def main():
    if len(sys.argv) < 2:
        print('Uso: python inspect_papers.py "nome sorgente 1" ["nome sorgente 2" ...]')
        return

    for source_name in sys.argv[1:]:
        print(f"\n{'='*70}\n{source_name}\n{'='*70}")
        detailed_rows, aggregated = lit_pipeline.process_source(source_name, "anthropic", None)

        print(f"\nRisultato aggregato: classification_final={aggregated.get('classification_final')}, "
              f"redshift_final={aggregated.get('redshift_final')}, "
              f"n_papers_reviewed={aggregated.get('n_papers_reviewed')}, "
              f"n_extractions_verified={aggregated.get('n_extractions_verified')}")

        verified = [r for r in detailed_rows if r.get("quote_verified")]
        print(f"\nDettaglio delle {len(verified)} estrazioni verificate (bibcode, anno, "
              f"classificazione, quote):")
        for r in sorted(verified, key=lambda r: r.get("year") or 0, reverse=True):
            print(f"\n  bibcode: {r.get('bibcode')} ({r.get('year')})")
            print(f"  pub: {r.get('pub')}")
            print(f"  classificazione estratta: {r.get('spectral_classification')}")
            print(f"  redshift estratto: {r.get('redshift')} ({r.get('redshift_type')})")
            print(f"  quote: {r.get('quote')}")

    totals = llm_extract.get_usage_totals()
    cost = llm_extract.estimate_cost_usd("anthropic", "claude-sonnet-5")
    print(f"\n[COSTO] {totals['n_calls']} chiamate LLM -> stima: ${cost:.3f}" if cost else "")


if __name__ == "__main__":
    main()
