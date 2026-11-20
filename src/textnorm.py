"""Text normalisation shared by training, decoding and scoring.

One normaliser is used everywhere (training targets AND WER scoring) so that
every model is trained on, and judged against, identical text.

Steps (strict mode):
  1. Unicode NFC.
  2. Drop zero-width characters (ZWSP, ZWNJ, ZWJ, WORD JOINER, BOM).
  3. Devanagari digits -> ASCII digits.
  4. Lower-case (only affects Latin letters).
  5. Every character that is not a Letter (L*), Mark (M*), Number (N*) or
     whitespace becomes a space. This removes the danda (U+0964/0965), all
     punctuation and symbols while keeping Devanagari vowel signs / virama
     (category M*).
  6. Collapse whitespace.

Lenient mode (secondary metric only) additionally:
  * removes the nukta (U+093C), so क़ == क, ज़ == ज, ऩ == न
  * maps chandrabindu (U+0901) to anusvara (U+0902)
This is reported as a second column because annotators and models disagree
about nukta / chandrabindu a lot and it can dominate small WER differences.
"""
from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_DEVA_DIGITS = {ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}
_WS = re.compile(r"\s+")

NUKTA = "़"
CHANDRABINDU = "ँ"
ANUSVARA = "ं"


def normalize(text: str | None, lenient: bool = False) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = t.translate(_ZERO_WIDTH)
    t = t.translate(_DEVA_DIGITS)
    t = t.lower()
    t = "".join(
        ch if (unicodedata.category(ch)[0] in "LMN" or ch.isspace()) else " "
        for ch in t
    )
    if lenient:
        # NFD first so precomposed nukta letters (e.g. U+0929) split too.
        t = unicodedata.normalize("NFD", t).replace(NUKTA, "")
        t = t.replace(CHANDRABINDU, ANUSVARA)
        t = unicodedata.normalize("NFC", t)
    return _WS.sub(" ", t).strip()


if __name__ == "__main__":  # tiny self-check
    assert normalize("नमस्ते, दुनिया।") == "नमस्ते दुनिया"
    assert normalize("आज १२ बजे  हैं॥") == "आज 12 बजे हैं"
    assert normalize("ज़रूर") == "ज़रूर"
    assert normalize("ज़रूर", lenient=True) == normalize("जरूर", lenient=True)
    assert normalize("गाँव", lenient=True) == normalize("गांव", lenient=True)
    assert normalize("Hello, World!") == "hello world"
    assert normalize(None) == ""
    print("textnorm OK")
