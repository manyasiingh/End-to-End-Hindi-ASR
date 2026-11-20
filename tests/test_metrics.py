"""Unit tests: our WER/CER must match jiwer; normaliser behaves. Run: python tests/test_metrics.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jiwer
from metrics import bootstrap_ci, per_utterance, rate, wer_cer, breakdown
from textnorm import normalize

random.seed(0)
WORDS = "आज मौसम बहुत अच्छा है हम घर जा रहे हैं कल स्कूल बंद रहेगा पानी गर्म करो".split()


def rand_sent(n):
    return " ".join(random.choice(WORDS) for _ in range(n))


def mutate(s):
    w = s.split()
    for _ in range(random.randint(0, 3)):
        op = random.choice("sdi")
        if op == "s" and w:
            w[random.randrange(len(w))] = random.choice(WORDS)
        elif op == "d" and len(w) > 1:
            w.pop(random.randrange(len(w)))
        else:
            w.insert(random.randrange(len(w) + 1), random.choice(WORDS))
    return " ".join(w)


refs = [rand_sent(random.randint(3, 12)) for _ in range(300)]
hyps = [mutate(r) for r in refs]
hyps[5] = ""  # empty hypothesis must work

wer, cer = wer_cer(refs, hyps)
j_wer = jiwer.wer(refs, hyps)
j_cer = jiwer.cer(refs, hyps)
assert abs(wer - j_wer) < 1e-9, (wer, j_wer)
assert abs(cer - j_cer) < 1e-9, (cer, j_cer)

a = per_utterance(refs, hyps, "word")
b = breakdown(a)
assert abs(b["sub"] + b["del"] + b["ins"] - b["err"]) < 1e-9
o = jiwer.process_words(refs, hyps)
assert abs((o.substitutions + o.deletions + o.insertions) / (o.hits + o.substitutions + o.deletions) - wer) < 1e-9
lo, hi = bootstrap_ci(a, 200)
assert lo <= rate(a) <= hi, (lo, rate(a), hi)

# empty reference dropped, not crashing
assert len(per_utterance(["", "क ख"], ["x", "क"], "word")) == 1

# normaliser
for s, want in [("नमस्ते, दुनिया।", "नमस्ते दुनिया"), ("१२ बजे॥", "12 बजे"), ("A.B  c", "a b c")]:
    assert normalize(s) == want, (s, normalize(s), want)
assert normalize("क़िला", lenient=True) == normalize("किला", lenient=True)
assert normalize("ऩ", lenient=True) == normalize("न", lenient=True)
print(f"metrics OK  (WER {wer:.4f} == jiwer {j_wer:.4f}, CER {cer:.4f} == jiwer {j_cer:.4f})")
