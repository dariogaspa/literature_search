# literature_search

Pipeline di ricerca in letteratura scientifica (ADS/SIMBAD/LLM) per il recupero automatico di
redshift, classificazione spettrale e angolo di vista di sorgenti AGN del catalogo Fermi-LAT
(5FGL e successori), pensata per le sorgenti prive di una misura affidabile nei cataloghi
automatici (NED, SIMBAD, VizieR).

Il progetto nasce sulla falsariga di quanto proposto da L. Foschini per il catalogo FL16Y:
invece di limitarsi ai valori riportati nei cataloghi automatici, la pipeline recupera i paper
scientifici associati a ciascuna sorgente e ne estrae con un LLM (Claude o GPT, intercambiabili)
i dati richiesti, quando esplicitamente riportati nel testo — con verifica automatica delle
citazioni per limitare le allucinazioni.

Include anche un modulo di validazione che confronta l'output della pipeline con il catalogo
già pubblicato da Foschini et al. 2022 (Universe, 8, 587), usato come ground truth indipendente.

## Documentazione

- `literature_pipeline_documentation.docx` — documentazione completa della pipeline principale
  (architettura, schema di estrazione, meccanismi anti-allucinazione, aggregazione, note tecniche)
- `foschini_validation_documentation.docx` — documentazione del modulo di validazione

## Struttura

| File | Responsabilità |
|---|---|
| `ads_client.py` | Interrogazione NASA ADS; bibliografia SIMBAD via TAP/ADQL |
| `fulltext_fetch.py` | Recupero testo/tabelle dei paper (arXiv/ar5iv); controlli anti-contaminazione |
| `vizier_table_lookup.py` | Recupero tabelle dati integrali dei paper depositate su VizieR |
| `llm_extract.py` | Estrazione strutturata via LLM con verifica delle citazioni |
| `aggregate.py` | Aggregazione deterministica multi-paper per sorgente |
| `pipeline.py` | Orchestrazione end-to-end con checkpoint/resume |
| `diagnose.py` | Diagnostica passo-passo su una singola sorgente |
| `fetch_foschini_catalog.py` | Download del catalogo ufficiale Foschini+2022 da VizieR |
| `reproduce_foschini.py` | Confronto pipeline vs. catalogo Foschini (validazione) |
| `inspect_source.py` | Ispezione rapida della riga completa del catalogo Foschini per una sorgente |
| `inspect_papers.py` | Rilancio mirato su singole sorgenti per ispezione dettagliata per-paper |

## Setup

```bash
pip install -r requirements.txt

export ADS_TOKEN=...          # https://ui.adsabs.harvard.edu/user/settings/token
export ANTHROPIC_API_KEY=...  # se si usa --provider anthropic
export OPENAI_API_KEY=...     # se si usa --provider openai
```

## Utilizzo

```bash
# Pipeline principale
python pipeline.py --provider anthropic --limit 20   # test su 20 sorgenti
python pipeline.py --provider anthropic              # intero batch

# Diagnostica su una singola sorgente
python diagnose.py "nome sorgente"

# Validazione contro il catalogo Foschini
python fetch_foschini_catalog.py
python reproduce_foschini.py --limit 25 --seed 42

# Ispezione di un disaccordo specifico
python inspect_source.py "nome sorgente"
python inspect_papers.py "nome sorgente"
```

Vedi la documentazione completa per i dettagli su schema di estrazione, regole di
aggregazione, meccanismi anti-allucinazione e note tecniche/limitazioni note.
