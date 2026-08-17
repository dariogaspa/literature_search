"""
fetch_foschini_catalog.py
--------------------------
Scarica il catalogo pubblicato Foschini+2022 (Universe 8, 587) da VizieR
(J/other/Univ/8.587) e lo salva come CSV pulito, pronto per essere usato
da reproduce_foschini.py come ground truth "ufficiale" (2980 sorgenti),
al posto della tabella di appunti informale usata nel primo test.

Uso:
    python fetch_foschini_catalog.py
"""

from astroquery.vizier import Vizier
import pandas as pd

Vizier.ROW_LIMIT = -1  # nessun limite: servono tutti i ~2980 record

CATALOG_ID = "J/other/Univ/8.587"
OUTPUT_CSV = "foschini_catalog_full.csv"

print(f"Scarico {CATALOG_ID} da VizieR (puo' volerci un minuto per ~2980 righe)...")
catalogs = Vizier.get_catalogs(CATALOG_ID)

print(f"\nTabelle trovate: {len(catalogs)}")
for i, t in enumerate(catalogs):
    print(f"  [{i}] {t.meta.get('name', '?')} - {len(t)} righe, colonne: {t.colnames}")

# la tabella principale (catalog.dat) e' quella con ~2980 righe e le colonne
# RevCl/Class/z; la scegliamo automaticamente cercando la colonna RevCl
main_table = None
notes_table = None
for t in catalogs:
    if "RevCl" in t.colnames:
        main_table = t
    elif "Notes" in t.colnames:
        # notes.dat: tabella separata con le note testuali estese, collegata
        # a catalog.dat tramite 4FGLname. Il campo "Notes" DENTRO catalog.dat
        # e' spesso vuoto: e' solo un marcatore (es. "*") che rimanda a questa
        # tabella, quindi va recuperata a parte per avere il testo vero.
        notes_table = t

if main_table is None:
    print("\n[ATTENZIONE] Non ho trovato una tabella con colonna 'RevCl'. "
          "Controlla l'elenco sopra e adatta lo script (potrebbe essere "
          "necessario un nome tabella esplicito, es. catalogs['J/other/Univ/8.587/catalog']).")
else:
    df = main_table.to_pandas()

    # decodifica eventuali colonne bytes -> str (comune con astroquery/astropy)
    def _decode_df(d):
        for col in d.columns:
            if d[col].dtype == object:
                d[col] = d[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
                d[col] = d[col].str.strip() if d[col].dtype == object else d[col]
        return d

    df = _decode_df(df)

    if notes_table is not None:
        notes_df = _decode_df(notes_table.to_pandas())
        print(f"\nTabella note estese trovata: {len(notes_df)} righe, colonne: {notes_df.columns.tolist()}")
        # la colonna "Notes" di catalog.dat e' quasi sempre vuota (solo un
        # marcatore); la sovrascriviamo con il testo vero da notes.dat dove
        # disponibile, tenendo il nome di colonna "Notes" per compatibilita'
        # con inspect_source.py
        df = df.drop(columns=["Notes"], errors="ignore")
        df = df.merge(notes_df[["4FGLname", "Notes"]], on="4FGLname", how="left")
    else:
        print("\n[ATTENZIONE] Nessuna tabella di note estese trovata insieme al catalogo "
              "principale: il campo Notes restera' vuoto per la maggior parte delle sorgenti.")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSalvate {len(df)} righe in {OUTPUT_CSV}")
    print(f"Colonne: {df.columns.tolist()}")
    print(f"\nDistribuzione RevCl:")
    print(df["RevCl"].value_counts(dropna=False))
    if "Notes" in df.columns:
        n_with_notes = df["Notes"].notna().sum()
        print(f"\nSorgenti con nota estesa: {n_with_notes}/{len(df)}")
