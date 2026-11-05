"""
Verificador: compara la respuesta final del modelo con el ground truth.

Debe aceptar formas equivalentes (criterio iii, Robustez):
  - Números: "40", "40.0", "cuarenta", "40 estudiantes", "4,0" vs "4.0". Los enteros deben
    coincidir exactamente; los no enteros aceptan redondeo a 2 decimales o la tolerancia relativa.
  - Listas / conjuntos: orden irrelevante salvo que `ordered=True` (rankings); separadores
    (coma, punto y coma, salto de línea, viñetas) normalizados; mayúsculas y acentos ignorados.
  - Booleanos: "sí"/"no", "true"/"false", "verdadero"/"falso".
  - Tablas pequeñas: filas separadas por líneas y columnas por '|', tab o coma. Sin `ordered`,
    la comparación es por conjunto de filas.

Y debe rechazar intentos de engaño:
  - Varias respuestas candidatas ("40 o 41"), rangos, respuestas vacías o "ver resultado".
  - Respuestas que contienen el valor correcto rodeado de texto adicional con otros números.

`answer_type` es el del dataset: number | text | boolean | list | table.
Se aceptan los alias scalar_number y scalar_text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional, Sequence

_ANSWER_ALIASES = {"scalar_number": "number", "scalar_text": "text"}
_MAX_ANSWER_CHARS = 2_000

_NUM_RE = re.compile(
    r"[+-]?(?:"
    r"\d{1,3}(?:\.\d{3})+,\d+"
    r"|\d{1,3}(?:,\d{3})+\.\d+"
    r"|\d{1,3}(?:\.\d{3})+"
    r"|\d{1,3}(?:,\d{3})+"
    r"|\d+(?:[.,]\d+)?"
    r")"
)

_ES_UNITS = {
    "cero": 0, "uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiun": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
}
_ES_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
_EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORDS = set(_ES_UNITS) | set(_EN_UNITS) | {"mil", "thousand", "hundred", "y", "and"}

_YES = {"si", "yes", "true", "verdadero", "verdadera", "correcto", "correcta", "1"}
_NO = {"no", "false", "falso", "falsa", "incorrecto", "incorrecta", "0"}
_UNKNOWN = {"no se", "no lo se", "nose", "n a", "na", "ver resultado", "ninguno", "ninguna", "no hay"}
_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "es", "son", "fue", "del", "de", "al", "se", "su",
    "en", "que", "por", "para", "con", "respuesta", "resultado", "paciente", "pacientes",
    "medico", "medicos", "doctor", "doctora", "departamento", "nombre", "diagnostico",
    "total", "hay", "fueron", "registraron", "equivale", "equivalen", "pesos", "clp",
    "estudiantes", "estudiante", "consultas", "consulta",
    "aprox", "aproximadamente", "unos", "unas", "cerca", "redondeado", "redondeada",
}


@dataclass
class VerificationResult:
    is_correct: bool
    normalized_prediction: Any = None
    normalized_target: Any = None
    reason: str = ""


def normalize_number(text: str) -> Optional[float]:
    """Convierte texto a número. None si no hay exactamente un número (dígitos o palabras)."""
    if not isinstance(text, str) or not text.strip():
        return None
    matches = list(_NUM_RE.finditer(text))
    if len(matches) > 1:
        return None
    if len(matches) == 1:
        rest = text[: matches[0].start()] + " " + text[matches[0].end() :]
        if _contains_number_word(rest):
            return None
        return _parse_numeric_token(matches[0].group(0))
    return _parse_number_words(normalize_text(text))


def normalize_text(text: str) -> str:
    """Minúsculas, sin acentos, sin puntuación, espacios colapsados."""
    if text is None:
        return ""
    decomposed = unicodedata.normalize("NFD", str(text))
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    stripped = stripped.lower().replace("ñ", "n")
    stripped = re.sub(r"[^\w\s]", " ", stripped, flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip()


def verify(
    prediction: str,
    target: Any,
    answer_type: str,
    tolerance: float = 0.0,
    ordered: bool = False,
) -> VerificationResult:
    """Compara `prediction` (texto del modelo) con `target` según `answer_type`.

    `tolerance` es una tolerancia relativa adicional para números no enteros (0 = solo
    coincidencia al centésimo, coherente con `ROUND(..., 2)` del ground truth).
    """
    kind = _ANSWER_ALIASES.get(answer_type, answer_type)
    if not isinstance(prediction, str) or not prediction.strip():
        return VerificationResult(False, reason="respuesta vacía")
    if len(prediction) > _MAX_ANSWER_CHARS:
        return VerificationResult(False, reason="respuesta demasiado larga")
    if kind == "number":
        return _verify_number(prediction, target, tolerance)
    if kind == "text":
        return _verify_text(prediction, target)
    if kind == "boolean":
        return _verify_boolean(prediction, target)
    if kind == "list":
        return _verify_list(prediction, target, tolerance, ordered)
    if kind == "table":
        return _verify_table(prediction, target, tolerance, ordered)
    return VerificationResult(False, reason=f"tipo de respuesta desconocido: {answer_type}")


def _verify_number(prediction: str, target: Any, tolerance: float) -> VerificationResult:
    parsed = normalize_number(prediction)
    if parsed is None or isinstance(target, bool) or not isinstance(target, (int, float)):
        return VerificationResult(False, parsed, target, "número no interpretable")
    ok = _numbers_close(parsed, float(target), tolerance, integer_target=float(target).is_integer())
    return VerificationResult(ok, parsed, target, "" if ok else "número distinto")


def _verify_text(prediction: str, target: Any) -> VerificationResult:
    got = normalize_text(prediction)
    exp = normalize_text(target)
    if not exp:
        return VerificationResult(False, got, exp, "objetivo vacío")
    if got == exp:
        return VerificationResult(True, got, exp)
    if _contains_phrase(got, exp) and _remainder_is_filler(got, exp):
        return VerificationResult(True, got, exp)
    return VerificationResult(False, got, exp, "texto distinto")


def _verify_boolean(prediction: str, target: Any) -> VerificationResult:
    parsed = _parse_bool(prediction)
    expected = bool(target)
    if parsed is None:
        return VerificationResult(False, None, expected, "booleano no interpretable")
    return VerificationResult(parsed is expected, parsed, expected, "" if parsed is expected else "booleano distinto")


def _verify_list(prediction: str, target: Any, tolerance: float, ordered: bool) -> VerificationResult:
    if not isinstance(target, (list, tuple)):
        return VerificationResult(False, reason="objetivo no es lista")
    items = _split_items(prediction)
    if len(items) != len(target):
        return VerificationResult(False, items, list(target), "cantidad de elementos distinta")
    if ordered:
        ok = all(_cell_equal(pred, exp, tolerance) for pred, exp in zip(items, target))
    else:
        ok = _match_unordered(items, list(target), tolerance)
    return VerificationResult(ok, items, list(target), "" if ok else "lista distinta")


def _verify_table(prediction: str, target: Any, tolerance: float, ordered: bool) -> VerificationResult:
    if not isinstance(target, (list, tuple)) or not target:
        return VerificationResult(False, reason="objetivo no es tabla")
    rows = _split_table(prediction)
    if len(rows) == len(target) + 1 and _looks_like_header(rows[0], target[0]):
        rows = rows[1:]
    if len(rows) != len(target):
        return VerificationResult(False, rows, list(target), "cantidad de filas distinta")
    if ordered:
        ok = all(_row_equal(pred, exp, tolerance) for pred, exp in zip(rows, target))
    else:
        ok = _match_unordered(rows, list(target), tolerance, row=True)
    return VerificationResult(ok, rows, list(target), "" if ok else "tabla distinta")


def _numbers_close(pred: float, target: float, tolerance: float, integer_target: bool) -> bool:
    if integer_target:
        return abs(pred - target) < 1e-6
    if round(pred, 2) == round(target, 2):
        return True
    return tolerance > 0 and abs(pred - target) <= tolerance * max(abs(target), 1.0)


def _parse_numeric_token(token: str) -> Optional[float]:
    t = token.strip().replace(" ", "")
    if re.fullmatch(r"[+-]?\d+", t):
        return float(t)
    has_dot, has_comma = "." in t, "," in t
    try:
        if has_dot and has_comma:
            if t.rfind(",") > t.rfind("."):
                t = t.replace(".", "").replace(",", ".")
            else:
                t = t.replace(",", "")
            return float(t)
        if has_comma and re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+", t):
            return float(t.replace(",", ""))
        if has_dot and re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})+", t):
            return float(t.replace(".", ""))
        t = t.replace(",", ".")
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", t):
            return float(t)
    except ValueError:
        return None
    return None


def _contains_number_word(text: str) -> bool:
    words = set(_ES_UNITS) | set(_EN_UNITS) | {"mil", "thousand", "hundred"}
    return any(tok in words for tok in normalize_text(text).split())


def _parse_number_words(normalized: str) -> Optional[float]:
    tokens = [t for t in normalized.split() if t not in _STOPWORDS]
    if not tokens or any(t not in _NUMBER_WORDS for t in tokens):
        return None
    if len(tokens) == 1:
        word = tokens[0]
        if word in _ES_UNITS:
            return float(_ES_UNITS[word])
        if word in _EN_UNITS:
            return float(_EN_UNITS[word])
        return None
    if len(tokens) == 3 and tokens[1] in ("y", "and") and tokens[0] in _ES_TENS and tokens[2] in _ES_UNITS:
        return float(_ES_TENS[tokens[0]] + _ES_UNITS[tokens[2]])
    return None


def _parse_bool(text: str) -> Optional[bool]:
    norm = normalize_text(text)
    if not norm or norm in _UNKNOWN or norm.startswith("no se"):
        return None
    tokens = norm.split()
    content = [t for t in tokens if t not in _STOPWORDS or t in _YES or t in _NO]
    yes = [t for t in content if t in _YES]
    no = [t for t in content if t in _NO]
    other = [t for t in content if t not in _YES and t not in _NO]
    if other or (yes and no) or (not yes and not no):
        return None
    if len(yes) + len(no) != 1:
        return None
    return bool(yes)


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _remainder_is_filler(text: str, phrase: str) -> bool:
    remainder = f" {text} ".replace(f" {phrase} ", " ").split()
    return all(tok in _STOPWORDS for tok in remainder)


def _split_items(text: str) -> list[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"[\n;]|(?:\s*\|\s*)", cleaned)
    if len(chunks) == 1:
        chunks = re.split(r",", chunks[0])
    if len(chunks) == 1 and re.search(r"\s+y\s+", chunks[0], flags=re.IGNORECASE):
        chunks = re.split(r"\s+y\s+", chunks[0], flags=re.IGNORECASE)
    items = []
    for chunk in chunks:
        chunk = re.sub(r"^[\s\-\*\u2022]+", "", chunk.strip())
        chunk = re.sub(r"^\d+[\.\)]\s*", "", chunk).strip()
        if chunk:
            items.append(chunk)
    return items


def _split_table(text: str) -> list[list[str]]:
    rows = []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line or re.fullmatch(r"\|?[\s:\-]+\|?", line):
            continue
        line = re.sub(r"^[\-\*\u2022]\s+", "", line)
        if "|" in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
        elif "\t" in line:
            cells = [c.strip() for c in line.split("\t")]
        else:
            cells = [c.strip() for c in line.split(",")]
        cells = [c for c in cells if c]
        if not cells or all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def _looks_like_header(row: Sequence[str], sample: Sequence[Any]) -> bool:
    """Una fila de encabezado no calza con los tipos de la fila de datos (p.ej. texto donde va un número)."""
    if len(row) != len(sample):
        return False
    saw_mismatch = False
    for cell, expected in zip(row, sample):
        if isinstance(expected, bool):
            continue
        if isinstance(expected, (int, float)) and normalize_number(cell) is None:
            saw_mismatch = True
    return saw_mismatch


def _cell_equal(pred: str, target: Any, tolerance: float) -> bool:
    if isinstance(target, bool):
        return _parse_bool(pred) is target
    if isinstance(target, (int, float)):
        parsed = normalize_number(pred)
        return parsed is not None and _numbers_close(parsed, float(target), tolerance, float(target).is_integer())
    return normalize_text(pred) == normalize_text(target)


def _row_equal(pred: Sequence[str], target: Sequence[Any], tolerance: float) -> bool:
    return len(pred) == len(target) and all(_cell_equal(p, t, tolerance) for p, t in zip(pred, target))


def _match_unordered(preds: Sequence[Any], targets: Sequence[Any], tolerance: float, row: bool = False) -> bool:
    used: set[int] = set()
    for target in targets:
        found = False
        for i, pred in enumerate(preds):
            if i in used:
                continue
            ok = _row_equal(pred, target, tolerance) if row else _cell_equal(pred, target, tolerance)
            if ok:
                used.add(i)
                found = True
                break
        if not found:
            return False
    return True
