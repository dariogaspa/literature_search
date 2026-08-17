"""
reproduce_foschini.py
----------------------
Fa girare la nostra pipeline di ricerca in letteratura sulle sorgenti che
Luigi Foschini ha gia' classificato a mano (foschini_comparison.csv), e
confronta il risultato con la sua classificazione, mappando il nostro
output (testo libero, estratto letteralmente dai paper) nel suo schema a
8 categorie tramite regole esplicite e deterministiche (nessun LLM in
questo passaggio, per restare tracciabili).

Schema di Foschini (definizioni fornite da lui):
  FSRQ  = Flat-Spectrum Radio Quasar
  BLLAC = BL Lac Object
  NLS1  = Narrow-Line Seyfert 1 Galaxy
  SEY   = Seyfert galaxy (Type 1, 2, intermedia, LINER)
  MIS   = Misaligned Jetted AGN
  CLAGN = Changing-look AGN
  AMB   = Ambiguous
  UNCL  = Unclassified

Uso:
    export ADS_TOKEN=...
    export ANTHROPIC_API_KEY=...
    python reproduce_foschini.py --limit 25 --seed 42
"""

import argparse
import os
import random
import re
import time
import pandas as pd

import pipeline as lit_pipeline  # riusa process_source() gia' scritto
import llm_extract

OFFICIAL_CATALOG_CSV = "foschini_catalog_full.csv"     # da fetch_foschini_catalog.py
NOTES_CATALOG_CSV = "foschini_comparison.csv"           # appunti informali (fallback)
OUTPUT_CSV = "foschini_reproduction_results.csv"
DETAILED_OUTPUT_CSV = "foschini_reproduction_detailed.csv"

# Il catalogo ufficiale usa "BLLAC" per esteso; il resto delle sigle e'
# gia' identico al vocabolario che usiamo internamente (minuscolo).
REVCL_TO_SHORT = {
    "FSRQ": "fsrq", "BLLAC": "bll", "NLS1": "nls1", "SEY": "sey",
    "MIS": "mis", "CLAGN": "clagn", "AMB": "amb", "UNCL": "uncl",
}


def load_unified_input(source="auto"):
    """Carica il catalogo ufficiale VizieR se disponibile (fetch_foschini_catalog.py
    gia' eseguito), altrimenti ripiega sulla tabella di appunti informale.
    Ritorna un DataFrame con colonne unificate: assoc, foschini_true,
    e se disponibili anche z_foschini, z_flag_foschini.

    source: "auto" (default, preferisce l'ufficiale se presente),
            "official" (forza il catalogo ufficiale, errore se assente),
            "informal" (forza gli appunti informali, es. per confronti
            "prima/dopo" sullo stesso campione gia' testato in precedenza,
            senza consumare token su sorgenti nuove mai viste)."""
    use_official = (source == "official") or (source == "auto" and os.path.exists(OFFICIAL_CATALOG_CSV))

    if use_official:
        if not os.path.exists(OFFICIAL_CATALOG_CSV):
            raise FileNotFoundError(
                f"{OFFICIAL_CATALOG_CSV} non trovato. Esegui prima fetch_foschini_catalog.py, "
                f"oppure usa --source informal per il campione di appunti."
            )
        print(f"[INFO] Uso il catalogo ufficiale VizieR: {OFFICIAL_CATALOG_CSV}")
        df = pd.read_csv(OFFICIAL_CATALOG_CSV)
        df = df.rename(columns={"Counterpt": "assoc"})
        df["foschini_true"] = df["RevCl"].map(REVCL_TO_SHORT).fillna(
            df["RevCl"].astype(str).str.lower()
        )
        if "z" in df.columns:
            df["z_foschini"] = df["z"]
        if "f_z" in df.columns:
            df["z_flag_foschini"] = df["f_z"]
        return df

    print(f"[INFO] Uso gli appunti informali: {NOTES_CATALOG_CSV}")
    df = pd.read_csv(NOTES_CATALOG_CSV)
    df["foschini_true"] = df["class_Foschini"].astype(str).str.lower()
    return df

# --- mappatura deterministica: testo libero estratto -> schema Foschini ---
# L'ordine conta: le regole piu' specifiche vanno controllate prima di
# quelle piu' generiche (es. "narrow-line seyfert" prima di "seyfert").
_MAPPING_RULES = [
    (r"narrow[\s-]?line seyfert|nls1|nlsy1", "nls1"),
    (r"changing[\s-]?look", "clagn"),
    (r"\bbzb\b|bl lac|\bbll\b|\b[lih]bl\b|\b[lih]b\b|\b[lih]sp\b", "bll"),
    # bzb = sigla Roma-BZCAT per BL Lac; lbl/ibl/hbl (o lb/ib/hb) e lsp/isp/hsp
    # (low/intermediate/high [synchrotron-]peaked) sono nomenclature diverse
    # per la stessa sottoclassificazione SED dei BL Lac: sono comunque BL Lac
    # ai fini del confronto con lo schema di Foschini.
    (r"\bbzq\b|flat[\s-]?spectrum radio quasar|\bfsrq\b", "fsrq"),  # bzq = sigla Roma-BZCAT per FSRQ
    (r"broad[\s-]?line seyfert|blsy1|seyfert|liner", "sey"),
    (r"radio galaxy|misaligned|wide[\s-]?angle[\s-]?tailed|\bwat\b|\bfr\s?i+\b|nelrg", "mis"),
    (r"ambiguous|uncertain|unclear|\bbzu\b", "amb"),  # bzu = sigla Roma-BZCAT per tipo incerto
]


def map_to_foschini_scheme(classification_text):
    """Ritorna una delle 8 categorie di Foschini (minuscolo), oppure 'uncl'
    se non c'e' classificazione, oppure 'altro/non mappato' se il testo
    estratto non corrisponde a nessuna regola nota (cosi' non forziamo un
    'amb' che non e' realmente giustificato)."""
    if not classification_text or str(classification_text).strip() == "":
        return "uncl"
    text = str(classification_text).lower()
    for pattern, label in _MAPPING_RULES:
        if re.search(pattern, text):
            return label
    return "altro/non mappato"


SAMPLE_FILE = "foschini_reproduction_sample.txt"


def select_sample(df, limit, seed, sample_file=None, save_sample=True):
    """
    Seleziona il campione di sorgenti da testare. Se sample_file esiste,
    lo ricarica da li' (garantendo lo STESSO identico campione tra run
    diversi, anche se il codice di filtro/campionamento cambia nel
    frattempo: il seed da solo non basta, perche' l'algoritmo di
    campionamento di pandas e' sensibile all'ordine/numero di righe in
    ingresso). Altrimenti campiona da df e salva l'elenco per i run
    successivi.
    """
    sample_file = sample_file or SAMPLE_FILE

    if os.path.exists(sample_file):
        print(f"[INFO] Ricarico il campione fisso da {sample_file} (per modificare il campione, "
              f"cancella questo file).")
        with open(sample_file) as f:
            names = [line.strip() for line in f if line.strip()]
        sample = df[df["assoc"].isin(names)].drop_duplicates(subset="assoc").reset_index(drop=True)
        # mantiene l'ordine del file, non quello del dataframe
        sample["_order"] = sample["assoc"].map({n: i for i, n in enumerate(names)})
        sample = sample.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        missing = set(names) - set(sample["assoc"])
        if missing:
            print(f"[ATTENZIONE] {len(missing)} sorgenti del campione fisso non trovate nel "
                  f"catalogo corrente (es. cambiato --source): {sorted(missing)[:5]}...")
        return sample

    with_truth = df[df["foschini_true"].notna() & (df["foschini_true"] != "nan")]
    if limit and limit < len(with_truth):
        sample = with_truth.sample(n=limit, random_state=seed).reset_index(drop=True)
    else:
        sample = with_truth.reset_index(drop=True)

    if save_sample:
        with open(sample_file, "w") as f:
            f.write("\n".join(sample["assoc"].astype(str)))
        print(f"[INFO] Campione salvato in {sample_file}: riusalo nei prossimi run per un "
              f"confronto prima/dopo stabile (cancella il file per campionarne uno nuovo).")

    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", choices=["auto", "official", "informal"], default="auto",
                         help="Quale catalogo usare come ground truth: 'official' (VizieR, "
                              "2980 sorgenti), 'informal' (appunti, 366 sorgenti, utile per "
                              "confronti prima/dopo sullo stesso campione gia' testato), "
                              "'auto' (default: preferisce l'ufficiale se presente).")
    parser.add_argument("--resample", action="store_true",
                         help="Ignora il campione fisso salvato e ne sceglie uno nuovo "
                              "(sovrascrive foschini_reproduction_sample.txt).")
    args = parser.parse_args()

    df = load_unified_input(source=args.source)
    if args.resample and os.path.exists(SAMPLE_FILE):
        os.remove(SAMPLE_FILE)
        print(f"[INFO] --resample: rimosso il campione fisso precedente ({SAMPLE_FILE}).")

    sample = select_sample(df, args.limit, args.seed)
    print(f"[INFO] Campione selezionato: {len(sample)} sorgenti (seed={args.seed})")

    results = []
    all_detailed_rows = []
    for idx, row in sample.iterrows():
        source_name = row["assoc"]
        foschini_true = row["foschini_true"]
        z_foschini = row.get("z_foschini")
        z_flag_foschini = row.get("z_flag_foschini")
        print(f"[{idx+1}/{len(sample)}] {source_name} (Foschini dice: {foschini_true})")

        try:
            detailed_rows, aggregated = lit_pipeline.process_source(source_name, args.provider, args.model)
        except Exception as e:
            print(f"  [ERRORE] {source_name}: {e}")
            detailed_rows = []
            aggregated = {"n_papers_reviewed": 0, "n_extractions_verified": 0,
                          "classification_final": None, "redshift_final": None}

        all_detailed_rows.extend(detailed_rows)

        our_mapped = map_to_foschini_scheme(aggregated.get("classification_final"))
        agree = (our_mapped == str(foschini_true).lower())

        z_ours = aggregated.get("redshift_final")
        z_agree = None
        if z_ours is not None and pd.notna(z_foschini):
            z_agree = abs(float(z_ours) - float(z_foschini)) < 0.01

        results.append({
            "assoc": source_name,
            "foschini_true": foschini_true,
            "our_classification_raw": aggregated.get("classification_final"),
            "our_classification_mapped": our_mapped,
            "agree": agree,
            "our_redshift_final": z_ours,
            "z_foschini": z_foschini,
            "z_flag_foschini": z_flag_foschini,
            "z_agree": z_agree,
            "n_papers_reviewed": aggregated.get("n_papers_reviewed", 0),
            "n_extractions_verified": aggregated.get("n_extractions_verified", 0),
            "n_llm_errors": aggregated.get("n_llm_errors", 0),
        })

        time.sleep(1.0)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    if all_detailed_rows:
        detailed_df = pd.DataFrame(all_detailed_rows)
        detailed_df.to_csv(DETAILED_OUTPUT_CSV, index=False)
        print(f"[INFO] Dettaglio per-paper salvato in {DETAILED_OUTPUT_CSV} "
              f"({len(detailed_df)} righe: sorgente x paper x chunk).")

    # --- riepilogo ---
    n = len(out_df)
    n_agree = out_df["agree"].sum()
    n_with_data = (out_df["n_extractions_verified"] > 0).sum()

    print(f"\n=== RIEPILOGO ===")
    print(f"Sorgenti confrontate: {n}")
    print(f"Con almeno un'estrazione verificata: {n_with_data}/{n}")
    print(f"Accordo con Foschini (su tutte, incluse quelle senza dati -> 'uncl'): "
          f"{n_agree}/{n} ({100*n_agree/n:.0f}%)")

    only_with_data = out_df[out_df["n_extractions_verified"] > 0]
    if len(only_with_data) > 0:
        n_agree_data = only_with_data["agree"].sum()
        print(f"Accordo con Foschini (solo dove abbiamo trovato almeno un dato): "
              f"{n_agree_data}/{len(only_with_data)} ({100*n_agree_data/len(only_with_data):.0f}%)")

    z_comparable = out_df[out_df["z_agree"].notna()]
    if len(z_comparable) > 0:
        n_z_agree = z_comparable["z_agree"].sum()
        print(f"Accordo sul redshift (|Δz|<0.01, dove entrambi disponibili): "
              f"{n_z_agree}/{len(z_comparable)} ({100*n_z_agree/len(z_comparable):.0f}%)")

    print("\nDisaccordi:")
    for _, r in out_df[~out_df["agree"]].iterrows():
        print(f"  {r['assoc']}: Foschini={r['foschini_true']}, noi={r['our_classification_mapped']} "
              f"(raw: {r['our_classification_raw']}, {r['n_papers_reviewed']} paper, "
              f"{r['n_extractions_verified']} verificate)")

    totals = llm_extract.get_usage_totals()
    cost = llm_extract.estimate_cost_usd(args.provider, args.model or "claude-sonnet-5")
    print(f"\n[COSTO] {totals['n_calls']} chiamate LLM -> stima: ${cost:.3f}" if cost else "")
    if totals["n_errors"] > 0:
        print(f"[ATTENZIONE] {totals['n_errors']} chiamate LLM fallite con errore: le statistiche "
              f"sopra possono essere sottostimate. Controlla la colonna n_llm_errors in "
              f"{OUTPUT_CSV} prima di trarre conclusioni.")


if __name__ == "__main__":
    main()
