"""
inspect_source.py
------------------
Stampa la riga completa del catalogo ufficiale di Foschini (compreso il
campo Notes) per una o piu' sorgenti, per capire velocemente su quale
base ha deciso una classificazione controversa - utile quando la nostra
pipeline trova un paper diverso e arriva a una conclusione diversa.

Uso:
    python inspect_source.py "TXS 1206+549" "B2 1100+30B"
"""

import sys
import pandas as pd

CATALOG_CSV = "foschini_catalog_full.csv"


def main():
    if len(sys.argv) < 2:
        print("Uso: python inspect_source.py \"nome sorgente 1\" [\"nome sorgente 2\" ...]")
        return

    df = pd.read_csv(CATALOG_CSV)

    for name in sys.argv[1:]:
        matches = df[df["Counterpt"].astype(str).str.strip() == name.strip()]
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        if matches.empty:
            print("  Nessuna corrispondenza esatta trovata. Sorgenti simili:")
            similar = df[df["Counterpt"].astype(str).str.contains(name.split()[0], case=False, na=False)]
            print(similar[["Counterpt"]].head(5).to_string(index=False))
            continue

        for _, row in matches.iterrows():
            print(f"  4FGLname:  {row.get('4FGLname')}")
            print(f"  z:         {row.get('z')} (flag f_z={row.get('f_z')})")
            print(f"  r_z (rif. redshift): {row.get('r_z')}")
            print(f"  Class (4FGL-DR2 originale): {row.get('Class')}")
            print(f"  RevCl (classificazione rivista Foschini): {row.get('RevCl')}")
            print(f"  Notes:\n    {row.get('Notes')}")


if __name__ == "__main__":
    main()
