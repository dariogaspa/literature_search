"""
llm_extract.py
--------------
Estrazione strutturata di redshift / classificazione / angolo di vista
da un chunk di testo (abstract o full text), con:

- schema JSON forzato (tool-use per Anthropic, function-calling/json
  mode per OpenAI) -> niente parsing fragile di testo libero
- temperature=0 per la massima riproducibilità (solo provider OpenAI: claude-sonnet-5
  ha deprecato questo parametro e lo gestisce internamente)
- verifica automatica della "quote": se l'LLM non fornisce una frase
  verbatim (o quasi) rintracciabile nel testo sorgente, l'estrazione
  viene scartata automaticamente. Questo è il controllo principale
  contro le allucinazioni.

Provider supportati: "anthropic", "openai". Stesso schema di output
per entrambi, così i risultati sono direttamente confrontabili.
"""

import os
import json
import difflib
import re

def _nullable(schema):
    """Campo opzionale: valore del tipo indicato oppure null. Uso anyOf
    invece di 'type': [tipo, 'null'] perché è il pattern JSON Schema più
    universalmente supportato dai validatori (compresi quelli usati dalle
    API di tool-use), mentre gli array di tipi danno risultati incoerenti
    a seconda dell'implementazione."""
    return {"anyOf": [schema, {"type": "null"}]}


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "redshift": _nullable({"type": "number"}),
        "redshift_type": _nullable({
            "type": "string",
            "enum": ["spectroscopic", "photometric", "tentative"],
        }),
        "spectral_classification": _nullable({"type": "string"}),
        "viewing_angle_deg": _nullable({"type": "number"}),
        "viewing_angle_method": _nullable({
            "type": "string",
            "enum": ["VLBI superluminal motion", "SED modeling", "other"],
        }),
        "quote": _nullable({"type": "string"}),
        "confidence": _nullable({
            "type": "string",
            "enum": ["explicit", "inferred_by_paper_authors"],
        }),
    },
    "required": [
        "redshift", "redshift_type", "spectral_classification",
        "viewing_angle_deg", "viewing_angle_method", "quote", "confidence",
    ],
}

SYSTEM_PROMPT = (
    "Sei un assistente per l'estrazione di dati astrofisici da testi "
    "scientifici. Estrai SOLO informazioni esplicitamente presenti nel "
    "testo fornito. Non dedurre, non inferire, non completare con "
    "conoscenza esterna. Se un campo non è presente nel testo, usa null. "
    "Il campo 'quote' deve essere una frase copiata ESATTAMENTE dal testo "
    "(verbatim, non parafrasata) che supporta i valori estratti."
)


def build_user_prompt(source_name, text_chunk):
    return (
        f"Sorgente: {source_name}\n\n"
        f"Testo (estratto da un paper scientifico):\n{text_chunk}\n\n"
        "Estrai le informazioni secondo lo schema JSON fornito."
    )


# ---------------------------------------------------------------------
# Provider: Anthropic
# ---------------------------------------------------------------------

def _call_anthropic(source_name, text_chunk, model="claude-sonnet-5"):
    import anthropic

    client = anthropic.Anthropic()  # legge ANTHROPIC_API_KEY da env

    tool = {
        "name": "record_extraction",
        "description": "Registra i dati estratti dal testo scientifico.",
        "input_schema": EXTRACTION_SCHEMA,
    }

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "record_extraction"},
        messages=[{"role": "user", "content": build_user_prompt(source_name, text_chunk)}],
    )

    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    for block in resp.content:
        if block.type == "tool_use":
            return block.input, usage
    return None, usage


# ---------------------------------------------------------------------
# Provider: OpenAI
# ---------------------------------------------------------------------

def _call_openai(source_name, text_chunk, model="gpt-4.1"):
    from openai import OpenAI

    client = OpenAI()  # legge OPENAI_API_KEY da env

    tools = [{
        "type": "function",
        "function": {
            "name": "record_extraction",
            "description": "Registra i dati estratti dal testo scientifico.",
            "parameters": EXTRACTION_SCHEMA,
        },
    }]

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(source_name, text_chunk)},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "record_extraction"}},
    )

    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    msg = resp.choices[0].message
    if msg.tool_calls:
        args = msg.tool_calls[0].function.arguments
        return json.loads(args), usage
    return None, usage


# Prezzi ufficiali (USD per milione di token, input/output). Sonnet 5 è in
# tariffa introduttiva fino al 31 agosto 2026, poi passa a $3/$15. Aggiorna
# questi valori se cambiano: https://www.anthropic.com/pricing
PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
}

PRICING_PER_MTOK = {
    ("anthropic", "claude-sonnet-5"): (2.00, 10.00),
    ("anthropic", "claude-haiku-4-5-20251001"): (1.00, 5.00),
    ("anthropic", "claude-opus-4-8"): (5.00, 25.00),
    ("openai", "gpt-4.1"): (2.00, 8.00),  # verifica sul sito OpenAI, non tracciato da Anthropic
}

# accumulatore globale semplice, usato da pipeline.py per stampare il costo
# stimato a fine run (o ai checkpoint)
_usage_totals = {"input_tokens": 0, "output_tokens": 0, "n_calls": 0, "n_errors": 0}


def get_usage_totals():
    return dict(_usage_totals)


def estimate_cost_usd(provider, model, input_tokens=None, output_tokens=None):
    """Stima il costo in USD dato il provider/modello e i token (usa il totale
    accumulato finora se input_tokens/output_tokens non sono specificati)."""
    if input_tokens is None:
        input_tokens = _usage_totals["input_tokens"]
    if output_tokens is None:
        output_tokens = _usage_totals["output_tokens"]

    rates = PRICING_PER_MTOK.get((provider, model))
    if rates is None:
        return None  # modello non in tabella: aggiungilo sopra per stimare il costo
    in_rate, out_rate = rates
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def verify_quote(quote, source_text, min_ratio=0.85, window_pad=40, min_quote_length=15):
    """
    Verifica che 'quote' sia effettivamente rintracciabile in 'source_text'.
    Prima prova un match esatto (case-insensitive, whitespace normalizzato);
    se fallisce, cerca la miglior finestra approssimata con difflib e
    accetta solo se il rapporto di similarità supera min_ratio.

    Le quote piu' corte di min_quote_length caratteri vengono SEMPRE
    considerate non verificate, anche se compaiono letteralmente nel
    testo: una citazione tipo "bcu" o "bll" (pochi caratteri) e'
    quasi garantito che compaia da qualche parte in un paper di
    classificazione blazar, quindi il match non dimostra che la citazione
    si riferisca davvero alla sorgente richiesta - non e' una verifica
    utile, e' un falso senso di sicurezza.

    Ritorna (bool verificata, str match_trovato_o_None).
    """
    if not quote:
        return False, None

    if len(quote.strip()) < min_quote_length:
        return False, None

    norm_quote = re.sub(r"\s+", " ", quote).strip().lower()
    norm_source = re.sub(r"\s+", " ", source_text).strip().lower()

    if norm_quote in norm_source:
        return True, quote

    # fuzzy match: scorri finestre di lunghezza simile alla quote
    qlen = len(norm_quote)
    if qlen == 0 or len(norm_source) < qlen:
        return False, None

    best_ratio = 0.0
    best_match = None
    step = max(1, qlen // 4)
    for i in range(0, max(1, len(norm_source) - qlen), step):
        window = norm_source[i:i + qlen + window_pad]
        ratio = difflib.SequenceMatcher(None, norm_quote, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = window

    if best_ratio >= min_ratio:
        return True, best_match
    return False, None


def extract_from_chunk(source_name, text_chunk, provider="anthropic", model=None, min_quote_length=15):
    """
    Esegue l'estrazione su un singolo chunk di testo e verifica la quote.
    Ritorna un dict con i campi dello schema + 'quote_verified' (bool)
    e 'provider'/'model' usati, oppure None in caso di errore/chunk vuoto.

    min_quote_length: soglia minima di lunghezza per considerare valida
    una citazione (vedi verify_quote). Il chiamante puo' abbassarla per i
    chunk che sono GIA' pre-filtrati sulla sorgente giusta (es. i blocchi
    tabella di fulltext_fetch.extract_relevant_table_rows /
    vizier_table_lookup: quella riga e' stata trovata cercando
    esplicitamente il nome della sorgente, quindi anche una citazione
    corta come "bll" o "IB" e' li' affidabile). Per chunk di prosa non
    filtrati, dove la stessa citazione corta potrebbe riferirsi a
    qualunque sorgente menzionata nel paper, si usa il default piu' severo.
    """
    if not text_chunk or not text_chunk.strip():
        return None

    call_fn = PROVIDERS.get(provider)
    if call_fn is None:
        raise ValueError(f"Provider non supportato: {provider}. Usa uno tra {list(PROVIDERS)}")

    kwargs = {"model": model} if model else {}
    try:
        result, usage = call_fn(source_name, text_chunk, **kwargs) if model else call_fn(source_name, text_chunk)
    except Exception as e:
        print(f"  [LLM/{provider}] errore su {source_name}: {type(e).__name__}: {e}")
        _usage_totals["n_errors"] += 1
        return None

    _usage_totals["input_tokens"] += usage.get("input_tokens", 0)
    _usage_totals["output_tokens"] += usage.get("output_tokens", 0)
    _usage_totals["n_calls"] += 1

    if result is None:
        return None

    quote = result.get("quote")
    if quote:
        verified, matched = verify_quote(quote, text_chunk, min_quote_length=min_quote_length)
    else:
        # Il modello non ha fornito nessuna citazione: e' la risposta onesta
        # quando non c'e' nulla di esplicito nel testo, NON un fallimento
        # della verifica. Va tenuto distinto dal caso "ha citato qualcosa
        # che non risulta nel testo", che e' la vera cattura anti-allucinazione.
        verified = False

    result["quote_verified"] = verified

    # Scarta i valori numerici/categorici SOLO se il modello ha fornito una
    # quote che non si e' rivelata verificabile: quello e' un segnale reale
    # di possibile allucinazione. Se invece non c'era nessuna quote da
    # verificare, i valori sono gia' tutti null di loro (il modello non ha
    # trovato nulla) e non serve sovrascrivere 'confidence'.
    if quote and not verified:
        for k in ("redshift", "spectral_classification", "viewing_angle_deg"):
            result[k] = None
        result["confidence"] = "unverified_quote"

    result["provider"] = provider
    result["model"] = model or ""
    return result
