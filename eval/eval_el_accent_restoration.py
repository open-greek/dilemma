#!/usr/bin/env python3
"""Accent restoration, typing slips, the enclitic second accent and
keystroke savings of the monotonic Modern Greek keyboard bundle, measured
on the held-out dev split.

Reads only what the keyboard ships: the word list
``build/hunspell/el_monotonic.dic`` (``export_el_monotonic.py``), looked up
by its accent-free key, and the next-word model ``build/lm/el_ngram.bin``
(``export_lm.py --lang el``), read with ``dilemma_build.gnlm_reader``,
which answers as the keyboard's reader does. The text is the dev split of
``build/el_corpus`` (``extract_el_monotonic_lm.py``), held out by document
group: neither exporter read it.

The dev documents are split once more by a hash of their id. The *tune*
half picks the model margin (a sample of it, ``--tune-sentences`` per
source); every reported number comes from the *test* half.

Measures
--------
* **Accent restoration.** Every word of a sentence is typed with no tonos
  and no dialytika (case kept: a sentence-initial capital stays), then
  restored word by word with ``el_accent.decide``: one candidate with the
  key is applied; several are decided by the model in left context (the
  restored words before it) when the best leads by ``--margin``, else the
  word stays as typed (policy ``keep``; policy ``top`` applies the best one
  when the typed letters are not a word). Replacing typed letters that are
  a word themselves (η by ή) needs the larger ``--word-margin``. A third
  run, ``keep_next_word``, also scores each candidate by the best
  candidate of the next typed word after it, as a keyboard can once the
  next word is typed. Then the enclitic second accent
  is added by ``el_accent.takes_enclitic_accent``. Words in capitals,
  placeholders and abbreviations are not typed. Reported per source:
  word accuracy (over all words, and over words whose spelling the list
  holds), the **damage rate** (words that were already right as typed, such
  as monosyllables, and came out wrong, over those words and over all
  words), the share left bare because the candidates were too close, the
  share with no candidate, the most frequent ambiguous groups and errors.
* **Typing slips.** On a sample of the test half, about one word in seven
  also gets one common slip before the accents are stripped: a spelling of
  /i/ for another (ι, η, υ, ει, οι), ο for ω, ε for αι, a doubled or
  undoubled consonant, or a neighboring key. Reported: how often the
  slipped word still has candidates (it became the key of another word, so
  restoration silently writes that word), how often it has none (left for
  typo repair), how often a same-sound lookup (``el_lexicon.sound_key``)
  would recover the intended word, and how often that lookup would change
  an unslipped word that has no candidate.
* **Enclitic second accent.** Over every dev occurrence of a word followed
  by an enclitic, whether the corpus spelling agrees with the syllable rule
  (``el_accent.enclitic_accent``: every proparoxytone host), by enclitic
  and by host class, and with the keyboard rule
  (``el_accent.takes_enclitic_accent``: μου, σου, μας and σας always, the
  article-like and preposition-like enclitics only at the end of a
  sentence). With ``--enclitic-evidence``, also a rule that follows the
  training split's own majority spelling of the host before that enclitic
  and the next word, or before that enclitic alone, when it has seen the
  pair at least ``--evidence-min`` times, and the keyboard rule otherwise.
  Corpus spellings are not all correct: writers often leave the second
  accent out.
* **Keystroke savings.** Each word is typed letter by letter without
  accents. Before the first letter the bar shows the first three next-word
  predictions (``LMReader.suggest``); after each letter, completions
  (``LMReader.suggestCompletions``); tapping the word when it appears costs
  one keystroke and inserts the space. Without the bar a word costs its
  characters plus the space. Four layouts: ``predictions only`` (the bar
  is used only before the first letter), ``completions only`` (only after
  a letter), ``three completions`` (every slot holds a completion) and
  ``commit-aware`` (when space would change the typed letters, one slot
  shows them and one the correction, so one slot remains, or none when an
  accent alternative takes it). Placeholders and abbreviations count in
  none of them.

Usage
-----
    python eval/eval_el_accent_restoration.py                 # about 6 minutes
    python eval/eval_el_accent_restoration.py --quick         # small samples, about 1 minute
    python eval/eval_el_accent_restoration.py --enclitic-evidence --workers 12
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dilemma_build import el_accent as A  # noqa: E402
from dilemma_build import el_corpus, el_lexicon as L  # noqa: E402
from dilemma_build import el_type_counts  # noqa: E402
from dilemma_build.el_ngram import (  # noqa: E402
    N_RESERVED, ONE_LETTER_CAPITAL_WORDS, UNK, Resolver, fold_enclitic_accent,
    load_dictionary)
from dilemma_build.gnlm_reader import GNLMReader, normalize  # noqa: E402

DEFAULT_DIC = ROOT / "build" / "hunspell" / "el_monotonic.dic"
DEFAULT_BIN = ROOT / "build" / "lm" / "el_ngram.bin"
DEFAULT_CORPUS = ROOT / "build" / "el_corpus"
DEFAULT_OUT = ROOT / "build" / "el_monotonic" / "accent_eval.json"
DEFAULT_MARGIN = 1.0
DEFAULT_WORD_MARGIN = 3.0
SWEEP = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
WORD_SWEEP = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, math.inf)
HALF_SEED = "accent-eval"
TYPO_SEED = 5151
TYPO_RATE = 0.15
APOSTROPHE = L.APOSTROPHE
LOG_ALPHA = math.log(0.4)
CONVERSATIONAL = ("tatoeba", "common_voice", "wiktionary_examples")
TOP_GROUPS = 25
# Accent pairs reported whatever their rank.
WATCHED_GROUPS = (("η", "ή"), ("που", "πού"), ("πως", "πώς"), ("ποτέ", "πότε"),
                  ("μια", "μία"), ("δυο", "δύο"), ("αλλά", "άλλα"),
                  ("παρά", "πάρα"), ("νόμου", "νομού"), ("χρόνια", "χρονιά"))
TOP_ERRORS = 40


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load_entries(path: Path) -> list[tuple[str, str]]:
    """``(spelling, rating)`` for every line of a word-list ``.dic``."""
    out = []
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            form, _, fields = line.rstrip("\n").partition("\t")
            rating = "X"
            for f in fields.split():
                if f.startswith("fr:"):
                    rating = f[3:]
            out.append((form, rating))
    return out


def doc_of(sentence_id: str) -> str:
    return sentence_id.rsplit(":", 1)[0]


def half_of(doc_id: str) -> str:
    bucket = el_corpus._bucket(f"{HALF_SEED}:{doc_id}")
    return "tune" if bucket < el_corpus.BUCKETS // 2 else "test"


def every_nth(items: list, limit: int | None) -> list:
    if not limit or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(k * step)] for k in range(limit)]


def dev_sentences(corpus_dir: Path, half: str | None,
                  limit: int | None) -> dict[str, list[list[str]]]:
    """Dev sentences per source from one half (None: both), every n-th one
    when capped, so a sample spans the whole file."""
    meta = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for source in sorted(meta["sources"]):
        sents = [toks for sid, toks in el_corpus.iter_corpus_sentences(
            corpus_dir, source, "dev")
            if half is None or half_of(doc_of(sid)) == half]
        out[source] = every_nth(sents, limit)
    return out


def typable(token: str) -> bool:
    """A word the simulation types: letters (and U+2019) only, not in
    capitals except the one-letter articles Ο, Η and Ή, not a placeholder
    or an abbreviation."""
    if not el_type_counts.is_word_token(token):
        return False
    core = token.replace(APOSTROPHE, "")
    if not core or any(c not in L.LETTERS for c in core):
        return False
    shape = L.case_shape(core)
    if shape == "mixed":
        return False
    if shape == "upper" and token not in ONE_LETTER_CAPITAL_WORDS:
        return False
    return True


# --------------------------------------------------------------------------
# Restoration
# --------------------------------------------------------------------------

class Restorer:
    """Word-by-word restoration with the keyboard's two artifacts."""

    def __init__(self, index: A.AccentIndex, lm: GNLMReader, *,
                 margin: float, policy: str, word_margin: float | None = None,
                 enclitics: bool = True, lookahead: bool = False):
        self.index = index
        self.lm = lm
        self.lookahead = lookahead
        self.margin = margin
        self.word_margin = margin if word_margin is None else word_margin
        self.policy = policy
        self.enclitics = enclitics

    def scorer(self, prev: str | None, curr: str | None):
        lm = self.lm
        pid, cid = lm.context_id(prev), lm.context_id(curr)

        def score(option: A.Option) -> float:
            best = -math.inf
            for s in option.spellings:
                wid = lm.id_case_insensitive(s)
                sc = (lm.unigram_score(lm.oov_count(s)) if wid is None
                      else lm.score_id(wid, pid, cid))
                best = max(best, sc)
            return best
        return score

    @staticmethod
    def context(out: list[str], i: int) -> tuple[str | None, str]:
        curr = out[i - 1] if i >= 1 else "<s>"
        prev = out[i - 2] if i >= 2 else ("<s>" if i == 1 else None)
        return prev, curr

    def with_next(self, score, curr: str | None, next_typed: str):
        """``score`` plus the best score of the next typed word's
        candidates after (``curr``, candidate): the decision waits for one
        more word, as the keyboard must for the article-like enclitics."""
        lm = self.lm
        nopts = self.index.options(next_typed)
        if not nopts:
            return score
        nexts = []
        for o in nopts:
            for sp in o.spellings:
                wid = lm.id_case_insensitive(sp)
                nexts.append((wid, sp))
        pid = lm.context_id(curr)

        def combined(option: A.Option) -> float:
            left = score(option)
            if left == -math.inf:
                return left
            cid = lm.id_case_insensitive(option.spellings[0])
            best = -math.inf
            for wid, sp in nexts:
                sc = (lm.unigram_score(lm.oov_count(sp)) if wid is None
                      else lm.score_id(wid, pid, cid))
                best = max(best, sc)
            return left + best if best > -math.inf else left
        return combined

    def decide(self, typed: str, out: list[str], i: int,
               next_typed: str | None = None) -> A.Decision:
        opts = self.index.options(typed)
        if len(opts) < 2:
            return A.decide(typed, opts, None, margin=self.margin,
                            policy=self.policy)
        prev, curr = self.context(out, i)
        score = self.scorer(prev, curr)
        if self.lookahead and next_typed is not None:
            score = self.with_next(score, curr, next_typed)
        return A.decide(typed, opts, score,
                        margin=self.margin, word_margin=self.word_margin,
                        policy=self.policy)

    def restore(self, typed: list[str | None], tokens: list[str]):
        """``typed[i]`` is what was typed for ``tokens[i]``, or None for a
        token that is not typed (it stays as written). Returns the output
        tokens, the decisions and the positions that got the enclitic
        accent."""
        out: list[str] = []
        decisions: list[A.Decision | None] = []
        for i, t in enumerate(typed):
            if t is None:
                out.append(tokens[i])
                decisions.append(None)
                continue
            nxt = typed[i + 1] if i + 1 < len(typed) else None
            d = self.decide(t, out, i, nxt)
            out.append(d.output)
            decisions.append(d)
        added = set()
        if self.enclitics:
            for i in range(len(out) - 1):
                if decisions[i] is None or typed[i + 1] is None:
                    continue
                enc = L.lower_el(typed[i + 1])
                if enc not in A.ENCLITICS:
                    continue
                nxt = tokens[i + 2] if i + 2 < len(tokens) else None
                if not A.takes_enclitic_accent(enc, nxt):
                    continue
                acc = A.enclitic_accent(out[i])
                if acc:
                    out[i] = acc
                    added.add(i)
        return out, decisions, added


def gold_listed(token: str, spellings: set[str]) -> bool:
    w = fold_enclitic_accent(token)
    return w in spellings or L.lower_el(w) in spellings


def group_name(d: A.Decision) -> str:
    return "/".join(sorted(d.options))


class AccentStats:
    def __init__(self):
        self.c = Counter()
        self.kinds = Counter()
        self.kinds_right = Counter()
        self.groups: dict[str, Counter] = defaultdict(Counter)
        self.group_out: dict[str, Counter] = defaultdict(Counter)
        self.errors = Counter()
        self.damage = Counter()
        self.unlisted_changes = Counter()

    def add(self, tokens, typed, out, decisions, added, spellings):
        c = self.c
        for i, d in enumerate(decisions):
            if d is None:
                continue
            gold = tokens[i]
            right = out[i] == gold
            listed = gold_listed(gold, spellings)
            bare_right = typed[i] == gold
            c["words"] += 1
            c["right"] += right
            c["listed"] += listed
            c["listed_right"] += listed and right
            c["bare_right"] += bare_right
            c["bare_right_listed"] += bare_right and listed
            if bare_right and not right:
                # Damage is measured on words the list holds. A word it
                # lacks that was typed as written and changed is usually a
                # corpus misspelling (απο), Ancient or Katharevousa (μοι,
                # αι) or a prefix cut from a hyphenated compound (αντι-);
                # those are counted apart.
                if listed:
                    c["damage"] += 1
                    self.damage[(gold, out[i], d.kind)] += 1
                else:
                    c["unlisted_changed"] += 1
                    self.unlisted_changes[(gold, out[i], d.kind)] += 1
            kind = d.kind
            self.kinds[kind] += 1
            self.kinds_right[kind] += right
            gold_double = L.tonos_count(gold) >= 2
            c["gold_double"] += gold_double
            c["gold_double_right"] += gold_double and right
            if i in added:
                c["enclitic_added"] += 1
                c["enclitic_added_right"] += right
            if len(d.options) >= 2:
                g = group_name(d)
                gc = self.groups[g]
                gc["n"] += 1
                gc["right"] += right
                gc["kept"] += kind == "kept"
                gc["listed"] += listed
                gc["listed_right"] += listed and right
                self.group_out[g][(L.lower_el(fold_enclitic_accent(gold)),
                                   L.lower_el(out[i]))] += 1
            if not right:
                self.errors[(L.lower_el(gold), L.lower_el(out[i]), kind)] += 1

    def merge(self, other: "AccentStats"):
        self.c.update(other.c)
        self.kinds.update(other.kinds)
        self.kinds_right.update(other.kinds_right)
        for k, v in other.groups.items():
            self.groups[k].update(v)
        for k, v in other.group_out.items():
            self.group_out[k].update(v)
        self.errors.update(other.errors)
        self.damage.update(other.damage)
        self.unlisted_changes.update(other.unlisted_changes)

    def summary(self) -> dict:
        c = self.c
        w = c["words"] or 1
        b = c["bare_right_listed"] or 1
        lst = c["listed"] or 1
        return {
            "words": c["words"],
            "accuracy": c["right"] / w,
            "accuracy_listed": c["listed_right"] / lst,
            "listed_share": c["listed"] / w,
            "bare_right_share": c["bare_right"] / w,
            "bare_right_listed": c["bare_right_listed"],
            "damage": c["damage"],
            "damage_rate": c["damage"] / b,
            "damage_per_word": c["damage"] / w,
            "unlisted_changed": c["unlisted_changed"],
            "kept_share": self.kinds["kept"] / w,
            "none_share": self.kinds["none"] / w,
            "kinds": {k: {"n": v, "right": self.kinds_right[k]}
                      for k, v in sorted(self.kinds.items())},
            "gold_double": c["gold_double"],
            "gold_double_right": c["gold_double_right"],
            "enclitic_added": c["enclitic_added"],
            "enclitic_added_right": c["enclitic_added_right"],
        }

    @staticmethod
    def _group_row(name: str, gc: Counter, outcomes: Counter) -> dict:
        n = gc["n"] or 1
        return {"group": name, "n": gc["n"], "accuracy": gc["right"] / n,
                "kept_share": gc["kept"] / n,
                "outcomes": [[gold, out, k] for (gold, out), k in
                             outcomes.most_common(6)]}

    def top_groups(self, n: int = TOP_GROUPS) -> list[dict]:
        return [self._group_row(g, gc, self.group_out[g]) for g, gc in
                sorted(self.groups.items(), key=lambda kv: -kv[1]["n"])[:n]]

    def watched_groups(self) -> list[dict]:
        """The ``WATCHED_GROUPS`` pairs, each summed over every group whose
        candidates include both spellings."""
        rows = []
        for pair in WATCHED_GROUPS:
            gc, outcomes = Counter(), Counter()
            for g, c in self.groups.items():
                if set(pair) <= set(g.split("/")):
                    gc.update(c)
                    outcomes.update(self.group_out[g])
            rows.append(self._group_row("/".join(pair), gc, outcomes))
        return rows

    def top_errors(self, n: int = TOP_ERRORS) -> list:
        return [[g, o, k, cnt] for (g, o, k), cnt in self.errors.most_common(n)]

    def top_damage(self, n: int = TOP_ERRORS) -> list:
        return [[g, o, k, cnt] for (g, o, k), cnt in self.damage.most_common(n)]

    def top_unlisted_changes(self, n: int = TOP_ERRORS) -> list:
        return [[g, o, k, cnt] for (g, o, k), cnt in
                self.unlisted_changes.most_common(n)]


def run_accent(restorer: Restorer, dev: dict[str, list[list[str]]],
               spellings: set[str]) -> dict[str, AccentStats]:
    out = {}
    for source, sents in dev.items():
        st = AccentStats()
        for toks in sents:
            typed = [A.strip_marks(t) if typable(t) else None for t in toks]
            res, decisions, added = restorer.restore(typed, toks)
            st.add(toks, typed, res, decisions, added, spellings)
        out[source] = st
    return out


def pooled(stats: dict[str, AccentStats], sources) -> AccentStats:
    total = AccentStats()
    for s in sources:
        if s in stats:
            total.merge(stats[s])
    return total


# --------------------------------------------------------------------------
# Typing slips
# --------------------------------------------------------------------------

I_SPELLINGS = ("ει", "οι", "ι", "η", "υ")
DOUBLED = "λμνρσττπκ"
LAYOUT_ROWS = ("ςερτυθιοπ", "ασδφγηξκλ", "ζχψωβνμ")
NEIGHBORS = {}
for _row in LAYOUT_ROWS:
    for _k, _c in enumerate(_row):
        NEIGHBORS[_c] = [_row[j] for j in (_k - 1, _k + 1) if 0 <= j < len(_row)]
VOWEL_LETTERS = frozenset("αεηιουω")


def _slips(word: str) -> dict[str, list[tuple[int, int, str]]]:
    """Every slip of each kind that applies to a lowercase bare word, as
    ``(start, end, replacement)``. Digraphs are respected: the ι of αι and
    the υ of ου are never touched on their own."""
    out: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    n = len(word)
    i = 0
    while i < n:
        pair = word[i:i + 2]
        if pair in ("ει", "οι"):
            for r in I_SPELLINGS:
                if r != pair:
                    out["i_spelling"].append((i, i + 2, r))
            i += 2
            continue
        if pair == "αι":
            out["e_ai"].append((i, i + 2, "ε"))
            i += 2
            continue
        if pair in ("ου", "αυ", "ευ", "ηυ", "υι"):
            i += 2
            continue
        c = word[i]
        if c in "ιηυ":
            for r in I_SPELLINGS:
                if r != c:
                    out["i_spelling"].append((i, i + 1, r))
        elif c == "ο":
            out["o_omega"].append((i, i + 1, "ω"))
        elif c == "ω":
            out["o_omega"].append((i, i + 1, "ο"))
        elif c == "ε":
            out["e_ai"].append((i, i + 1, "αι"))
        i += 1
    for i in range(n - 1):
        if word[i] == word[i + 1] and word[i] in DOUBLED:
            out["consonant_doubling"].append((i, i + 2, word[i]))
    for i in range(1, n - 1):
        if (word[i] in DOUBLED and word[i - 1] in VOWEL_LETTERS
                and word[i + 1] in VOWEL_LETTERS):
            out["consonant_doubling"].append((i, i + 1, word[i] * 2))
    for i, c in enumerate(word):
        for r in NEIGHBORS.get(c, ()):
            if r == "ς" and i != n - 1:
                continue
            if c == "ς":
                continue
            out["neighbor_key"].append((i, i + 1, r))
    return out


def make_slip(typed: str, rng: random.Random) -> tuple[str, str] | None:
    """One random slip in a bare typed word (case kept on the first
    letter), as ``(slipped, kind)``, or None when none applies."""
    if APOSTROPHE in typed or len(typed) < 2:
        return None
    low = L.lower_el(typed)
    slips = _slips(low)
    if not slips:
        return None
    kind = rng.choice(sorted(slips))
    start, end, rep = rng.choice(slips[kind])
    new = low[:start] + rep + low[end:]
    if new.endswith("σ"):
        new = new[:-1] + "ς"
    if "ς" in new[:-1]:
        new = new[:-1].replace("ς", "σ") + new[-1]
    if typed[0] != low[0]:
        new = new[0].upper() + new[1:]
    return new, kind


def apply_case(typed: str, spelling: str) -> str:
    if typed and typed[0].isupper() and spelling and spelling[0].islower():
        return spelling[0].upper() + spelling[1:]
    return spelling


class SoundIndex:
    """The word list by :func:`el_lexicon.sound_key`, for the same-sound
    lookup a typo repair would start from."""

    def __init__(self, entries):
        by: dict[str, dict[str, int]] = defaultdict(dict)
        for s, r in entries:
            low = L.lower_el(s)
            by[L.sound_key(s)][low] = max(by[L.sound_key(s)].get(low, 0),
                                         L.BUCKET_RANK.get(r, 0))
        self.by = by

    def options(self, typed: str) -> tuple[A.Option, ...]:
        found = self.by.get(L.sound_key(typed), {})
        return tuple(sorted((A.Option(low, (low,), r) for low, r in found.items()),
                            key=lambda o: (-o.rank, o.lower)))


def sound_repair(restorer: Restorer, sound: SoundIndex, typed: str,
                 out: list[str], i: int) -> str | None:
    opts = sound.options(typed)
    if not opts:
        return None
    if len(opts) == 1:
        return apply_case(typed, opts[0].spellings[0])
    prev, curr = restorer.context(out, i)
    score = restorer.scorer(prev, curr)
    scores = [score(o) for o in opts]
    if A.gap_of(scores) < restorer.margin or A.gap_of(scores) <= 0:
        return None
    best = max(range(len(opts)), key=lambda k: (scores[k], opts[k].rank, -k))
    return apply_case(typed, opts[best].spellings[0])


def run_slips(restorer: Restorer, sound: SoundIndex,
              dev: dict[str, list[list[str]]], spellings: set[str]) -> dict:
    rng = random.Random(TYPO_SEED)
    c = Counter()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    examples = Counter()
    clean = AccentStats()
    for source, sents in dev.items():
        for toks in sents:
            typed: list[str | None] = []
            slipped: dict[int, str] = {}
            for i, t in enumerate(toks):
                if not typable(t):
                    typed.append(None)
                    continue
                bare = A.strip_marks(t)
                if gold_listed(t, spellings) and rng.random() < TYPO_RATE:
                    s = make_slip(bare, rng)
                    if s is not None:
                        typed.append(s[0])
                        slipped[i] = s[1]
                        continue
                typed.append(bare)
            out, decisions, added = restorer.restore(typed, toks)
            # Words without a slip in sentences that have slips: does the
            # changed context hurt them?
            if slipped:
                mask = [None if (d is None or i in slipped) else d
                        for i, d in enumerate(decisions)]
                clean.add(toks, typed, out, mask, added, spellings)
            for i, d in enumerate(decisions):
                if d is None:
                    continue
                kind = slipped.get(i)
                if kind is None:
                    if d.kind == "none":
                        c["unslipped_none"] += 1
                        rep = sound_repair(restorer, sound, typed[i], out, i)
                        if rep is not None and rep != toks[i]:
                            c["unslipped_none_repaired"] += 1
                    continue
                k = by_kind[kind]
                k["n"] += 1
                c["slipped"] += 1
                if d.kind == "none":
                    k["none"] += 1
                    rep = sound_repair(restorer, sound, typed[i], out, i)
                    if rep is not None:
                        k["sound_changed"] += 1
                        k["sound_recovered"] += L.lower_el(rep) == L.lower_el(
                            fold_enclitic_accent(toks[i]))
                else:
                    k["real_word"] += 1
                    examples[(kind, L.lower_el(toks[i]), L.lower_el(out[i]))] += 1
    total = Counter()
    for k in by_kind.values():
        total.update(k)
    return {
        "rate": TYPO_RATE, "seed": TYPO_SEED,
        "slipped_words": c["slipped"],
        "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
        "all_kinds": dict(total),
        "unslipped_none": c["unslipped_none"],
        "unslipped_none_repaired": c["unslipped_none_repaired"],
        "unslipped_words_in_slipped_sentences": clean.summary(),
        "real_word_examples": [[k, g, o, n] for (k, g, o), n in examples.most_common(30)],
    }


# --------------------------------------------------------------------------
# The enclitic second accent against the corpus
# --------------------------------------------------------------------------

def host_class(base: str) -> str:
    if A.enclitic_accent(base) is not None:
        return "proparoxytone"
    if A.needs_synizesis_call(base):
        return "synizesis"
    return "other"


def enclitic_pairs(toks: list[str]):
    """``(base host, enclitic, corpus has the second accent, next token)``
    for every typed host with at least one tonos followed by an enclitic."""
    for i in range(len(toks) - 1):
        enc = toks[i + 1]
        if enc not in A.ENCLITICS:
            enc = enc.lower()
            if enc not in A.ENCLITICS:
                continue
        host = toks[i]
        if APOSTROPHE in host or not typable(host) or L.case_shape(host) == "upper":
            continue
        marks = L.tonos_count(host)
        if marks == 0 or marks > 2:
            continue
        base = fold_enclitic_accent(host)
        if L.syllable_count(base) < 2:
            continue
        nxt = toks[i + 2] if i + 2 < len(toks) else None
        yield base, enc, marks == 2, nxt


def _evidence_shard(task: tuple[str, int]) -> dict:
    path, threshold = task
    counts: dict[tuple[str, str], list[int]] = {}
    for doc in el_corpus.iter_shard_documents(Path(path)):
        if threshold < el_corpus.BUCKETS and el_corpus.cap_bucket(doc["id"]) >= threshold:
            continue
        for sent in doc["s"]:
            if not _ENCLITIC_RE.search(sent):
                continue
            for base, enc, double, nxt in enclitic_pairs(sent.split(" ")):
                if host_class(base) == "other":
                    continue
                low = L.lower_el(base)
                for key in ((low, enc), (low, enc, next_key(nxt))):
                    v = counts.setdefault(key, [0, 0])
                    v[0 if double else 1] += 1
    return counts


def next_key(token: str | None) -> str:
    return "</s>" if token is None else L.lower_el(token)


_ENCLITIC_RE = re.compile(
    r"(?:^| )(?:" + "|".join(sorted(A.ENCLITICS)) + r")(?: |$)", re.IGNORECASE)


def training_evidence(corpus_dir: Path, workers: int) -> dict:
    """``(lowercase host, enclitic)`` and ``(lowercase host, enclitic,
    next token)`` to ``[with the second accent, without]`` over the capped
    training split (the model's training text), for hosts whose spelling
    leaves the decision open (proparoxytones and glide words)."""
    meta = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    tasks = []
    for source, entry in sorted(meta["sources"].items()):
        threshold = entry["cap"]["bucket_threshold"]
        for shard in entry["splits"]["train"]["shards"]:
            tasks.append((str(corpus_dir / source / shard["file"]), threshold))
    total: dict[tuple[str, str], list[int]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_evidence_shard, tasks):
            for k, (d, s) in part.items():
                v = total.setdefault(k, [0, 0])
                v[0] += d
                v[1] += s
    return total


def iter_dev(corpus_dir: Path):
    """``(source, tokens)`` for every dev sentence, streamed."""
    meta = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    for source in sorted(meta["sources"]):
        for _, toks in el_corpus.iter_corpus_sentences(corpus_dir, source, "dev"):
            yield source, toks


ENCLITIC_GROUPS = (("always", A.ALWAYS_ENCLITICS), ("article", A.ARTICLE_ENCLITICS),
                   ("preposition", A.PREPOSITION_ENCLITICS))


def enclitic_summary(cells_by_enclitic: dict) -> dict:
    """Per group of enclitics: host occurrences, how many the corpus
    writes with the second accent, how many the rule accents, the share of
    those the corpus agrees with (precision) and the share of the corpus's
    accents the rule finds (recall)."""
    out = {}
    for name, members in ENCLITIC_GROUPS:
        t = [0, 0, 0, 0]
        for enc in members:
            t = [a + b for a, b in zip(t, cells_by_enclitic.get(enc, (0, 0, 0, 0)))]
        added, accented = t[0] + t[1], t[0] + t[2]
        out[name] = {"hosts": sum(t), "corpus_accented": accented, "rule_adds": added,
                     "precision": t[0] / added if added else 0.0,
                     "recall": t[0] / accented if accented else 0.0,
                     "agreement": (t[0] + t[3]) / sum(t) if sum(t) else 0.0}
    return out


def run_enclitics(sentences, evidence: dict | None, evidence_min: int) -> dict:
    # [class][enclitic] -> [rule&double, rule&single, norule&double, norule&single]
    table: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    keyboard = defaultdict(lambda: [0, 0, 0, 0])
    with_evidence = defaultdict(lambda: [0, 0, 0, 0])
    evidence_used = Counter()
    by_source = defaultdict(lambda: [0, 0])
    misses = Counter()
    false_adds = Counter()
    for source, toks in sentences:
        for base, enc, double, nxt in enclitic_pairs(toks):
            cls = host_class(base)
            rule = cls == "proparoxytone"
            cell = (0 if double else 1) if rule else (2 if double else 3)
            table[cls][enc][cell] += 1
            kb = rule and A.takes_enclitic_accent(enc, nxt)
            keyboard[enc][(0 if double else 1) if kb else (2 if double else 3)] += 1
            if rule:
                by_source[source][0] += 1
                by_source[source][1] += double
            if rule and not double:
                false_adds[(L.lower_el(base), enc)] += 1
            if not rule and double:
                misses[(L.lower_el(base), enc, cls)] += 1
            if evidence is not None:
                pred = kb
                if cls != "other":
                    low = L.lower_el(base)
                    keys = [(low, enc)]
                    if enc not in A.ALWAYS_ENCLITICS:
                        keys.insert(0, (low, enc, next_key(nxt)))
                    for key in keys:
                        ev = evidence.get(key)
                        if ev and ev[0] + ev[1] >= evidence_min:
                            pred = ev[0] > ev[1]
                            evidence_used[f"{cls} {len(key) - 1}-word"] += 1
                            break
                with_evidence[enc][(0 if double else 1) if pred else
                                   (2 if double else 3)] += 1

    def agree(cells):
        n = sum(cells)
        return (cells[0] + cells[3]) / n if n else 0.0

    pure = defaultdict(lambda: [0, 0, 0, 0])
    for encs in table.values():
        for enc, v in encs.items():
            pure[enc] = [a + b for a, b in zip(pure[enc], v)]
    rules = {"syllable_rule": pure, "keyboard_rule": keyboard}
    if evidence is not None:
        rules["with_training_evidence"] = with_evidence

    out = {"cells": "[rule and corpus accented, rule and corpus unaccented, "
                    "no rule and corpus accented, no rule and corpus unaccented]",
           "by_class": {cls: {enc: {"cells": v, "agreement": agree(v)}
                              for enc, v in sorted(encs.items())}
                        for cls, encs in sorted(table.items())},
           "keyboard_rule": {enc: {"cells": v, "agreement": agree(v)}
                             for enc, v in sorted(keyboard.items())},
           "proparoxytone_corpus_accented_by_source": {
               s: {"n": n, "accented": d, "share": d / n if n else 0.0}
               for s, (n, d) in sorted(by_source.items())},
           "top_unaccented_proparoxytone_hosts": [
               [h, e, n] for (h, e), n in false_adds.most_common(30)],
           "top_accented_hosts_the_rule_misses": [
               [h, e, c, n] for (h, e, c), n in misses.most_common(30)],
           "summary": {name: enclitic_summary(cells) for name, cells in rules.items()}}
    if evidence is not None:
        out["with_training_evidence"] = {
            "min_pairs": evidence_min,
            "hosts_decided_by_evidence": dict(evidence_used),
            "by_enclitic": {enc: {"cells": v, "agreement": agree(v)}
                            for enc, v in sorted(with_evidence.items())}}
    return out


# --------------------------------------------------------------------------
# Keystroke savings
# --------------------------------------------------------------------------

class Completer:
    """``LMReader.suggestCompletions`` for a stem typed without accents in
    lowercase, with an index instead of the vocabulary scan. The typed
    marks rule keeps every candidate of such a stem, so the scan ranks by
    count, then spelling; :meth:`check` compares with the reader."""

    def __init__(self, lm: GNLMReader):
        self.lm = lm
        norm, sorted_norm, order = lm._normalized_index()
        self.sorted_norm = sorted_norm
        self.order = np.asarray(order, dtype=np.int64)
        self.rank = np.empty(lm.V, dtype=np.int64)
        self.rank[self.order] = np.arange(lm.V)
        self.counts = lm.counts.astype(np.int64)
        self.vocab = lm.vocab
        self.reserved = lm.reserved_ids
        self.cache: dict[str, list[int]] = {}

    def span(self, nstem: str) -> tuple[int, int]:
        lo = bisect.bisect_left(self.sorted_norm, nstem)
        hi = bisect.bisect_left(self.sorted_norm, nstem + "\U0010ffff")
        return lo, hi

    def by_count(self, nstem: str, lo: int, hi: int, need: int) -> list[int]:
        got = self.cache.get(nstem)
        if got is not None and (len(got) >= need or len(got) == hi - lo):
            return got
        ids = self.order[lo:hi]
        if len(ids) > max(need, 64):
            cut = max(need, 64)
            part = np.argpartition(-self.counts[ids], cut - 1)[:cut]
            floor = self.counts[ids[part]].min()
            ids = ids[self.counts[ids] >= floor]
        ranked = sorted((int(w) for w in ids if int(w) not in self.reserved),
                        key=lambda w: (-self.counts[w], self.vocab[w]))
        self.cache[nstem] = ranked
        return ranked

    def complete(self, pid, cid, nstem: str, limit: int) -> list[int]:
        lm = self.lm
        lo, hi = self.span(nstem)
        if lo >= hi:
            return []
        pool = []
        seen = set()
        rows = []
        if pid is not None and cid is not None:
            rows.append((lm.trigram_row(pid, cid), 0.0))
        if cid is not None:
            rows.append((lm.bigram_row(cid), LOG_ALPHA))
        for row, backoff in rows:
            if row is None:
                continue
            ids, q = row
            r = self.rank[ids]
            for j in np.flatnonzero((r >= lo) & (r < hi)).tolist():
                w = int(ids[j])
                if w in seen or w in self.reserved:
                    continue
                seen.add(w)
                pool.append((backoff + q[j] / 1024.0, int(self.counts[w]), self.vocab[w], w))
        need = limit
        ranked = self.by_count(nstem, lo, hi, limit + len(seen))
        for w in ranked:
            if w in seen:
                continue
            c = int(self.counts[w])
            pool.append((lm.unigram_score(c), c, self.vocab[w], w))
            need -= 1
            if need <= 0:
                break
        pool.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return [w for _, _, _, w in pool[:limit]]

    def check(self, samples: list[tuple[str | None, str | None, str]]) -> int:
        """How many of ``(prev, curr, stem)`` disagree with the reader."""
        bad = 0
        lm = self.lm
        for prev, curr, stem in samples:
            want = lm.suggest_completions(prev, curr, stem, 3)
            got = [self.vocab[w] for w in self.complete(
                lm.context_id(prev), lm.context_id(curr), normalize(stem), 3)]
            bad += want != got
        return bad


LAYOUTS = ("predictions_only", "completions_only", "three_completions",
           "commit_aware")


def run_keystrokes(restorer: Restorer, completer: Completer, resolver: Resolver,
                   entries: list[str], dev: dict[str, list[list[str]]]) -> dict:
    lm = restorer.lm
    results = {}
    shown_pred = Counter()
    shown_comp = Counter()
    samples = []
    queries = 0

    def name(i):
        return None if i is None else lm.vocab[i]

    for source, sents in dev.items():
        c = Counter()
        for toks in sents:
            ids: list[int | None] = []
            for i, t in enumerate(toks):
                full, _ = resolver.resolve(t, i == 0)
                ids.append(None if full == UNK else lm.index.get(entries[full - N_RESERVED]))
            history: list[str] = []
            for i, t in enumerate(toks):
                if not typable(t):
                    history.append(t)
                    continue
                n = len(t)
                c["words"] += 1
                c["baseline"] += n + 1
                pid = ids[i - 2] if i >= 2 else (lm.id_bos if i == 1 else None)
                cid = ids[i - 1] if i >= 1 else lm.id_bos
                target = ids[i]
                typed = A.strip_marks(t)
                preds = lm.suggest_ids(pid, cid, 3)
                if preds:
                    shown_pred[lm.vocab[preds[0]]] += 1
                pred_hit = target is not None and target in preds
                comp_at: dict[str, int | None] = {"three": None, "aware": None}
                if target is not None and not pred_hit:
                    for k in range(1, n):
                        stem = typed[:k]
                        nstem = normalize(stem)
                        comps = completer.complete(pid, cid, nstem, 3)
                        if k == 1 and comps:
                            shown_comp[lm.vocab[comps[0]]] += 1
                        queries += 1
                        if len(samples) < 400 and queries % 211 == 0:
                            samples.append((name(pid), name(cid), L.lower_el(stem)))
                        if comp_at["three"] is None and target in comps:
                            comp_at["three"] = k
                        if comp_at["aware"] is None:
                            slots = 3
                            d = restorer.decide(stem, history, len(history))
                            if d.output != stem:
                                slots = 0 if len(d.options) > 1 else 1
                            if target in comps[:slots]:
                                comp_at["aware"] = k
                        if comp_at["three"] is not None and comp_at["aware"] is not None:
                            break
                full_cost = n + 1
                c["predictions_only"] += 1 if pred_hit else full_cost
                three = comp_at["three"]
                aware = comp_at["aware"]
                c["completions_only"] += (three + 1) if three is not None else full_cost
                c["three_completions"] += 1 if pred_hit else (
                    (three + 1) if three is not None else full_cost)
                c["commit_aware"] += 1 if pred_hit else (
                    (aware + 1) if aware is not None else full_cost)
                c["prediction_hits"] += pred_hit
                c["completion_hits_three"] += (not pred_hit) and three is not None
                c["completion_hits_aware"] += (not pred_hit) and aware is not None
                c["in_vocab"] += target is not None
                history.append(t)
        results[source] = dict(c)

    def summarize(c):
        base = c.get("baseline", 0) or 1
        w = c.get("words", 0) or 1
        s = {"words": c.get("words", 0), "baseline_keystrokes": c.get("baseline", 0),
             "in_vocab_share": c.get("in_vocab", 0) / w,
             "prediction_hit_share": c.get("prediction_hits", 0) / w,
             "completion_hit_share_three": c.get("completion_hits_three", 0) / w,
             "completion_hit_share_aware": c.get("completion_hits_aware", 0) / w}
        for lay in LAYOUTS:
            s[f"savings_{lay}"] = 1 - c.get(lay, 0) / base
        return s

    out = {"per_source": {s: summarize(c) for s, c in sorted(results.items())}}
    total = Counter()
    for c in results.values():
        total.update(c)
    out["pooled"] = {"all": summarize(total)}
    conv = Counter()
    for s in CONVERSATIONAL:
        conv.update(results.get(s, {}))
    if conv:
        out["pooled"]["conversational"] = summarize(conv)
    out["most_shown_first_prediction"] = shown_pred.most_common(30)
    out["most_shown_first_completion_after_one_letter"] = shown_comp.most_common(30)
    out["completion_check"] = {"samples": len(samples),
                               "mismatches": completer.check(samples)}
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def json_float(x: float):
    return "never" if x == math.inf else x


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def print_accent(title: str, stats: dict[str, AccentStats]) -> None:
    print(f"\n{title}")
    print("| Source | Words | Accuracy | Accuracy, listed words | Damage rate | "
          "Damage per word | Left bare, ambiguous | No candidate |")
    print("|---|---|---|---|---|---|---|---|")
    rows = list(stats.items()) + [("all, pooled", pooled(stats, stats))]
    conv = [s for s in CONVERSATIONAL if s in stats]
    if conv:
        rows.append(("conversational, pooled", pooled(stats, conv)))
    for name, st in rows:
        s = st.summary()
        print(f"| {name} | {s['words']:,} | {pct(s['accuracy'])} | "
              f"{pct(s['accuracy_listed'])} | {pct(s['damage_rate'])} | "
              f"{pct(s['damage_per_word'])} | {pct(s['kept_share'])} | "
              f"{pct(s['none_share'])} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dic", default=str(DEFAULT_DIC))
    ap.add_argument("--bin", default=str(DEFAULT_BIN))
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help="Lead in natural log units the model needs to choose "
                         f"between candidates (default {DEFAULT_MARGIN}).")
    ap.add_argument("--word-margin", type=float, default=DEFAULT_WORD_MARGIN,
                    help="Lead needed to replace typed letters that are a word "
                         "themselves (η to ή); 'inf' never replaces them "
                         f"(default {DEFAULT_WORD_MARGIN}).")
    ap.add_argument("--policy", choices=A.POLICIES, default="keep")
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="Per source, every n-th test-half sentence up to this many "
                         "(default: all) for accent restoration.")
    ap.add_argument("--tune-sentences", type=int, default=3000,
                    help="Per source, tune-half sentences for the margin sweep "
                         "(0 skips it).")
    ap.add_argument("--slip-sentences", type=int, default=3000,
                    help="Per source, test-half sentences for typing slips (0 skips).")
    ap.add_argument("--keystroke-sentences", type=int, default=3000,
                    help="Per source, test-half sentences for keystroke savings (0 skips).")
    ap.add_argument("--enclitic-evidence", action="store_true",
                    help="Also read the capped training split for the enclitic "
                         "evidence rule (about a minute with all cores).")
    ap.add_argument("--evidence-min", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="300 sentences per source everywhere, for a fast check.")
    args = ap.parse_args(argv)
    if args.quick:
        args.max_sentences = 300
        args.tune_sentences = min(args.tune_sentences, 300)
        args.slip_sentences = min(args.slip_sentences, 300)
        args.keystroke_sentences = min(args.keystroke_sentences, 300)

    t0 = time.time()
    corpus_dir = Path(args.corpus_dir)
    entries_r = load_entries(Path(args.dic))
    spellings = {s for s, _ in entries_r}
    index = A.AccentIndex(entries_r)
    lm = GNLMReader(Path(args.bin))
    print(f"word list: {len(entries_r):,} entries, {len(index):,} keys; "
          f"model: {lm.V:,} types ({time.time() - t0:.0f}s)")
    report: dict = {"inputs": {"dic": str(args.dic), "bin": str(args.bin),
                               "corpus_dir": str(corpus_dir),
                               "entries": len(entries_r), "keys": len(index),
                               "ambiguous_keys": sum(
                                   1 for v in index.by_key.values()
                                   if len({L.lower_el(s) for s, _ in v}) > 1)},
                    "margin": args.margin, "word_margin": json_float(args.word_margin),
                    "policy": args.policy}

    # Margin sweeps on the tune half.
    if args.tune_sentences:
        tune = dev_sentences(corpus_dir, "tune", args.tune_sentences)
        settings = [(policy, m, m, False) for policy in A.POLICIES for m in SWEEP]
        settings += [(args.policy, args.margin, wm, False) for wm in WORD_SWEEP]
        settings += [(args.policy, args.margin, wm, True) for wm in WORD_SWEEP]
        sweep = []
        for policy, m, wm, look in settings:
            st = run_accent(Restorer(index, lm, margin=m, word_margin=wm,
                                     policy=policy, lookahead=look), tune, spellings)
            s = pooled(st, st).summary()
            sweep.append({"policy": policy, "margin": m, "word_margin": json_float(wm),
                          "lookahead": look, **{
                k: s[k] for k in ("words", "accuracy", "accuracy_listed",
                                  "damage_rate", "damage_per_word", "kept_share")}})
        report["margin_sweep"] = {"half": "tune", "sentences_per_source": args.tune_sentences,
                                  "rows": sweep}
        print("\nMargin sweeps (tune half, pooled):")
        print("| Policy | Margin | Word margin | Next word | Accuracy | "
              "Accuracy, listed words | Damage rate | Left bare, ambiguous |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sweep:
            print(f"| {r['policy']} | {r['margin']} | {r['word_margin']} | "
                  f"{'yes' if r['lookahead'] else 'no'} | "
                  f"{pct(r['accuracy'])} | {pct(r['accuracy_listed'])} | "
                  f"{pct(r['damage_rate'])} | {pct(r['kept_share'])} |")
        print(f"({time.time() - t0:.0f}s)")

    test = dev_sentences(corpus_dir, "test", args.max_sentences)
    report["test_sentences"] = {k: len(v) for k, v in test.items()}
    accent = {}
    runs = [(p, p, False) for p in A.POLICIES]
    runs.append((args.policy + "_next_word", args.policy, True))
    for name, policy, look in runs:
        restorer = Restorer(index, lm, margin=args.margin,
                            word_margin=args.word_margin, policy=policy,
                            lookahead=look)
        st = run_accent(restorer, test, spellings)
        all_ = pooled(st, st)
        conv = pooled(st, [s for s in CONVERSATIONAL if s in st])
        accent[name] = {
            "per_source": {s: v.summary() for s, v in st.items()},
            "pooled": {"all": all_.summary(), "conversational": conv.summary()},
            "top_groups": all_.top_groups(),
            "top_groups_conversational": conv.top_groups(15),
            "watched_groups": all_.watched_groups(),
            "watched_groups_conversational": conv.watched_groups(),
            "top_errors": all_.top_errors(),
            "top_damage": all_.top_damage(),
            "top_unlisted_changes": all_.top_unlisted_changes(),
        }
        print_accent(f"Accent restoration, test half, policy {policy}, margin "
                     f"{args.margin}, word margin {args.word_margin}"
                     f"{', next word' if look else ''}:", st)
        print(f"({time.time() - t0:.0f}s)")
    report["accent"] = accent
    main_policy = accent[args.policy]
    print("\nMost frequent ambiguous groups (policy %s):" % args.policy)
    for g in main_policy["top_groups"][:15]:
        print(f"  {g['group']}: {g['n']:,} words, {pct(g['accuracy'])}% right, "
              f"{pct(g['kept_share'])}% left as typed; " +
              ", ".join(f"{a}->{b} {n}" for a, b, n in g["outcomes"][:4]))
    print("Most frequent damage:")
    for g, o, k, n in main_policy["top_damage"][:12]:
        print(f"  {g} -> {o} ({k}): {n}")
    print("Most frequent changes to unlisted words typed as written:")
    for g, o, k, n in main_policy["top_unlisted_changes"][:12]:
        print(f"  {g} -> {o} ({k}): {n}")
    print("Most frequent errors:")
    for g, o, k, n in main_policy["top_errors"][:20]:
        print(f"  {g} -> {o} ({k}): {n}")

    restorer = Restorer(index, lm, margin=args.margin,
                        word_margin=args.word_margin, policy=args.policy)

    if args.slip_sentences:
        slip_dev = {k: every_nth(v, args.slip_sentences) for k, v in test.items()}
        slips = run_slips(restorer, SoundIndex(entries_r), slip_dev, spellings)
        report["slips"] = slips
        a = slips["all_kinds"]
        n = a.get("n", 0) or 1
        print(f"\nTyping slips: {slips['slipped_words']:,} slipped words; "
              f"{pct(a.get('real_word', 0) / n)}% became another word's key, "
              f"{pct(a.get('none', 0) / n)}% have no candidate; same-sound lookup "
              f"recovers {pct(a.get('sound_recovered', 0) / n)}%")
        for kind, v in slips["by_kind"].items():
            kn = v.get("n", 0) or 1
            print(f"  {kind}: {v.get('n', 0):,}, real word {pct(v.get('real_word', 0) / kn)}%, "
                  f"none {pct(v.get('none', 0) / kn)}%, recovered "
                  f"{pct(v.get('sound_recovered', 0) / kn)}%")
        print(f"({time.time() - t0:.0f}s)")

    evidence = None
    if args.enclitic_evidence:
        evidence = training_evidence(corpus_dir, args.workers or 8)
        print(f"\ntraining evidence: {len(evidence):,} host and enclitic pairs "
              f"({time.time() - t0:.0f}s)")
    enc = run_enclitics(iter_dev(corpus_dir), evidence, args.evidence_min)
    report["enclitics"] = enc
    print("\nEnclitic second accent, whole dev split:")
    print("| Rule | Enclitics | Hosts | Corpus accented | Rule accents | "
          "Precision | Recall | Agreement |")
    print("|---|---|---|---|---|---|---|---|")
    for rule, groups in enc["summary"].items():
        for g, v in groups.items():
            print(f"| {rule} | {g} | {v['hosts']:,} | {v['corpus_accented']:,} | "
                  f"{v['rule_adds']:,} | {pct(v['precision'])} | {pct(v['recall'])} | "
                  f"{pct(v['agreement'])} |")
    print(f"({time.time() - t0:.0f}s)")

    if args.keystroke_sentences:
        ks_dev = {k: every_nth(v, args.keystroke_sentences) for k, v in test.items()}
        entries = load_dictionary(Path(args.dic))
        ks = run_keystrokes(restorer, Completer(lm), Resolver(entries), entries, ks_dev)
        report["keystrokes"] = ks
        print("\nKeystroke savings (test half sample):")
        print("| Set | Words | Predictions only | Completions only | "
              "Three completions | Commit-aware |")
        print("|---|---|---|---|---|---|")
        rows = list(ks["per_source"].items()) + [
            (f"{k}, pooled", v) for k, v in ks["pooled"].items()]
        for name, s in rows:
            print(f"| {name} | {s['words']:,} | " + " | ".join(
                pct(s[f"savings_{lay}"]) for lay in LAYOUTS) + " |")
        print("completion check:", ks["completion_check"])
        print("most shown first prediction:",
              ", ".join(f"{w} {n}" for w, n in ks["most_shown_first_prediction"][:20]))
        print("most shown first completion after one letter:",
              ", ".join(f"{w} {n}" for w, n in
                        ks["most_shown_first_completion_after_one_letter"][:20]))
    report["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWrote {out} in {report['seconds']}s")
    return report


if __name__ == "__main__":
    main()
