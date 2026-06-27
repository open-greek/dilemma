"""Per-work morphology reconciliation.

Reconcile a token's morphological analysis across several independent
annotation sources. The pass:

  1. votes the sources into a base ("chosen") reading while KEEPING every
     source's reading and the provenance (never overwrites, never silently
     drops a variant);
  2. auto-applies a few high-precision, morphologically *forced*
     corrections (these resolve a field with certainty);
  3. surfaces everything else as ``contested`` -- cross-source
     disagreement and the soft lexicon / syntax signals that, per this
     project's experience, over-fire and must not auto-apply; and
  4. optionally hands the contested set to an injected adjudicator (e.g. a
     local LLM), promoting a contested field to applied ONLY when the
     adjudicator AND a deterministic signal agree.

    from dilemma import reconcile_token, reconcile_work

    tok = reconcile_token(
        "ὑπέσχετο",
        [{"source": "a", "lemma": "ὑπισχνέομαι", "upos": "VERB",
          "feats": {"Person": "3", "Number": "Sing", "Voice": "Act",
                    "Tense": "Past", "Aspect": "Perf", "Mood": "Ind"}}],
    )
    tok.chosen["feats"]["Voice"]   # -> "Mid"  (forced: -χετο cannot be active)
    tok.contested                  # -> False  (the rule resolved it)

The module is pure: stdlib only, no lookup-DB query, no model load, no
torch / onnxruntime / mlx import. It is source-agnostic on purpose --
callers decode their own tagset into the generic ``{source, lemma, upos,
feats}`` reading shape before calling in, so nothing here is tied to one
corpus or tagging convention. Feature values follow the project's UD-style
encoding: aorist = Tense=Past + Aspect=Perf, imperfect = Tense=Past +
Aspect=Imp, present = Tense=Pres (no Aspect), perfect = Tense=Pres +
Aspect=Perf, pluperfect = Tense=Pqp; Voice in {Act, Mid, Pass, "Mid,Pass"}.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict


__all__ = [
    "Reading",
    "Provenance",
    "ReconciledToken",
    "reconcile_token",
    "reconcile_work",
    "vote_readings",
    "FEAT_FIELDS",
    "PERF_STEMS",
]


# ---------------------------------------------------------------------------
# Vocabulary / constants
# ---------------------------------------------------------------------------

# The morphological feature fields we vote on (lemma and upos are voted
# separately). Order is the canonical render order.
FEAT_FIELDS = ("Person", "Number", "Tense", "Aspect", "Mood", "Voice",
               "Gender", "Case")

_MEDIOPASSIVE = frozenset({"Mid", "Pass", "Mid,Pass"})

# Elision / apostrophe marks Homer (and the treebanks) use to mark an elided
# final vowel: straight + curly apostrophe, the modifier-letter apostrophe,
# the combining comma-above the treebanks attach to an elided consonant, the
# Greek koronis, and the psili. Some are spacing (survive accent stripping),
# some combining (removed by it), so both are handled explicitly.
_ELISION = "'’ʼ̓̔᾽᾿`´"
_ELISION_RE = re.compile("[" + re.escape(_ELISION) + "]$")
_ELISION_TAIL_RE = re.compile("[" + re.escape(_ELISION) + "]+$")

# Mediopassive personal endings (primary -μαι/-σαι/-ται/-νται and secondary
# -μην/-σο/-το/-ντο/-σθον/-σθην/-σθε/-μεθα). An active verb cannot carry one
# of these, so a full (non-elided) mediopassive ending on a Voice=Act token
# forces Voice=Mid. Elided endings (-τʼ/-ντʼ) are deliberately excluded here
# because -τʼ can also be the active 2pl -τε; the elision cases are handled,
# more cautiously, by the elided-present flagger.
_MID_ENDING_RE = re.compile(
    # -σαι is intentionally excluded: it collides with the active aorist
    # infinitive (λῦσαι), the fem-pl active participle (-ουσαι), and the active
    # aorist optative 3sg (γηθήσαι), so it is not a safe auto-apply ending.
    # Genuine -σαι middles (rare; aor mid imperative, perfect mp 2sg) fall
    # through to the contested flaggers / adjudication instead.
    "(?:μην|μεθα|σθην|σθον|σθε|νται|ντο|ται|το|μαι)"
    "[" + re.escape(_ELISION) + "]?$"
)

# Aorist middle SECONDARY endings (sigmatic -σα- variants reduce to these).
# Used to retag an aorist "Mid,Pass" as "Mid": a passive never ends in one of
# these (the aorist passive carries a -θη-/-η- formant with active-type
# endings), so the rule cannot mis-fire on a passive.
_AOR_MID_END_RE = re.compile("(?:σθην|σθον|σθε|μεθα|μην|ντο|το|σο)$")

# Default perfect / strong-aorist reduplication stems. These are generic
# Ancient-Greek facts (not corpus-specific): keyed by the tagged lemma, the
# value is a tuple of de-accented stem signatures that mark Aspect=Perf under
# the project encoding (aorist and perfect both -> Aspect=Perf). The signature
# is matched against the de-accented form, and gated on the lemma so it cannot
# fire on the present/imperfect of the same verb. Callers extend or replace
# the table via the ``perf_stems`` argument.
PERF_STEMS = {
    "ὄρνυμι": ("ορωρ",),                  # Attic-reduplicated perf (ὀρώρηται)
    "λανθάνω": ("λαθ", "λελαθ", "λελασ"),  # strong aor λαθ-, perf λελαθ-/λελασ-
}


# ---------------------------------------------------------------------------
# Normalization helpers (kept local so callers pay no package import cost)
# ---------------------------------------------------------------------------


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _deaccent(s: str) -> str:
    """Lower-case, strip combining diacritics (NFD, drop combining marks)."""
    nfd = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in nfd if not unicodedata.combining(c)).lower()


def _stem_tail(form: str) -> str:
    """De-accented form with any trailing (spacing) elision mark removed.

    Combining elision marks are already dropped by ``_deaccent``; this also
    strips a trailing spacing apostrophe so the final base consonant of an
    elided form is exposed (e.g. "πέτετʼ" -> "πετετ")."""
    return _ELISION_TAIL_RE.sub("", _deaccent(form))


def _is_elided(form: str) -> bool:
    return bool(_ELISION_RE.search(_nfc(form)))


def _is_verbal(chosen: dict) -> bool:
    if (chosen.get("upos") or "").upper() in {"VERB", "AUX"}:
        return True
    f = chosen.get("feats") or {}
    return any(f.get(k) for k in ("Person", "Tense", "Voice", "Mood"))


# ---------------------------------------------------------------------------
# Data structures (the output data contract)
# ---------------------------------------------------------------------------


@dataclass
class Reading:
    """One source's analysis of a token."""

    source: str
    lemma: str | None = None
    upos: str | None = None
    feats: dict = field(default_factory=dict)


@dataclass
class Provenance:
    """How a decision was reached. ``method`` is one of vote / rule:<name> /
    flag:<name> / adjudicator+<signal> / adjudicator:suggest."""

    method: str
    evidence: str = ""
    confidence: float = 0.0


@dataclass
class ReconciledToken:
    """Reconciled view of a single token (see module docstring)."""

    form: str
    chosen: dict                                    # {lemma, upos, feats}
    readings: list                                  # list[Reading]
    contested: bool = False
    contested_fields: list = field(default_factory=list)
    provenance: list = field(default_factory=list)  # list[Provenance]
    meta: dict = field(default_factory=dict)         # positional passthrough

    def to_dict(self) -> dict:
        """Emit the per-token data contract (JSON-ready). Any ``meta`` keys
        (line / col / idx / ...) are merged in at top level."""
        d = {
            "form": self.form,
            "chosen": self.chosen,
            "readings": [asdict(r) if isinstance(r, Reading) else dict(r)
                         for r in self.readings],
            "contested": self.contested,
            "contested_fields": list(self.contested_fields),
            "provenance": [asdict(p) if isinstance(p, Provenance) else dict(p)
                           for p in self.provenance],
        }
        d.update(self.meta or {})
        return d


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


def _coerce_readings(readings) -> list:
    out = []
    for r in readings:
        if isinstance(r, Reading):
            out.append(r)
        else:
            out.append(Reading(source=r.get("source", "?"),
                               lemma=r.get("lemma"),
                               upos=r.get("upos"),
                               feats=dict(r.get("feats") or {})))
    return out


def _majority(vals):
    """Return (winner, tie?) for a list of values. Empty -> (None, False)."""
    if not vals:
        return None, False
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    return winners[0], len(winners) > 1


def _field_values(rs, fld):
    if fld == "lemma":
        return [r.lemma for r in rs if r.lemma]
    if fld == "upos":
        return [r.upos for r in rs if r.upos]
    return [r.feats.get(fld) for r in rs if r.feats.get(fld)]


def vote_readings(readings):
    """Majority-vote a chosen reading and report per-field disagreement.

    ``readings`` are in caller PRIORITY order (first = primary). On a tie the
    primary's value is kept, but the field is still reported as a disagreement
    so the caller never silently picks one variant over another. Returns
    ``(chosen, disagree)`` where ``chosen`` is ``{lemma, upos, feats}`` and
    ``disagree`` is the list of fields the sources do not all agree on."""
    rs = _coerce_readings(readings)
    primary = rs[0] if rs else Reading(source="?")
    chosen = {"lemma": primary.lemma, "upos": primary.upos,
              "feats": dict(primary.feats)}
    disagree = []
    for key in ("lemma", "upos"):
        vals = _field_values(rs, key)
        win, tie = _majority(vals)
        if win is not None and not tie:
            chosen[key] = win
        if len(set(vals)) > 1:
            disagree.append(key)
    for fld in FEAT_FIELDS:
        vals = _field_values(rs, fld)
        if not vals:
            continue
        win, tie = _majority(vals)
        if win is not None and not tie:
            chosen["feats"][fld] = win
        if len(set(vals)) > 1:
            disagree.append(fld)
    return chosen, disagree


# ---------------------------------------------------------------------------
# High-precision rules (auto-apply; morphologically forced)
# ---------------------------------------------------------------------------
# Each returns (changes, name, evidence) or None.


def rule_mid_ending(form, feats):
    """A full mediopassive personal ending cannot be Voice=Act -> Mid.

    Infinitives are excluded: the active aorist infinitive ends in -σαι
    (λῦσαι, ποιῆσαι), colliding with the mediopassive 2sg -σαι, so an
    Act+Inf form is a legitimate active and must NOT be flipped to middle.
    No mediopassive infinitive ending (-σθαι) is in the table, so the guard
    loses no genuine middle. Infinitives carry VerbForm=Inf (no Mood field) in
    the UD encoding, so the guard checks VerbForm; Mood=Inf is also honoured for
    treebanks that tag the infinitive as a mood."""
    if feats.get("VerbForm") in ("Inf", "Part") or feats.get("Mood") == "Inf":
        return None
    if feats.get("Voice") != "Act":
        return None
    m = _MID_ENDING_RE.search(_deaccent(form))
    if not m:
        return None
    ending = m.group(0).rstrip(_ELISION)
    # σθε / σθον / σθην mark 2nd/3rd plural-or-dual; on a Number=Sing token they
    # are stem-final (ἄϊσθε, ὄλισθε = aorist 3sg active), not an inflectional
    # ending, so the rule must not fire.
    if ending in ("σθε", "σθον", "σθην") and feats.get("Number") == "Sing":
        return None
    # Middle and passive are syncretic (Mid,Pass) in the present / imperfect /
    # perfect; only the aorist keeps them distinct. So an aorist middle ending
    # is specifically Mid, and every other middle ending is the syncretic Mid,Pass.
    is_aor = feats.get("Tense") == "Past" and feats.get("Aspect") == "Perf"
    voice = "Mid" if is_aor else "Mid,Pass"
    return ({"Voice": voice}, "mid-ending",
            f"{form!r} ends in a mediopassive ending; Voice=Act is impossible")


def rule_perf_stem(form, lemma, feats, perf_stems):
    """A perfect / strong-aorist reduplication stem -> Aspect=Perf."""
    if feats.get("Aspect") == "Perf":
        return None
    # The pluperfect (Tense=Pqp) already encodes the perfect system and is
    # conventionally left Aspect-less in the UD treebanks (every gold Pqp has
    # no Aspect); adding Aspect=Perf there would diverge from the corpus.
    if feats.get("Tense") == "Pqp":
        return None
    sigs = (perf_stems or {}).get(lemma)
    if not sigs:
        return None
    ds = _deaccent(form)
    if any(sig in ds for sig in sigs):
        return ({"Aspect": "Perf"}, "perf-stem",
                f"{form!r} carries the {lemma} perfect/strong-aorist stem")
    return None


def rule_aorist_mid_voice(form, feats):
    """An aorist tagged Mid,Pass with a middle ending -> Mid.

    The syncretic "Mid,Pass" is correct ONLY where the middle and passive forms
    coincide: the present, imperfect, perfect, and pluperfect systems. The
    aorist (Tense=Past + Aspect=Perf in this encoding) keeps them distinct --
    the aorist passive carries a -θη-/-η- formant -- so a middle-ending aorist
    is specifically Mid, not the syncretic label. Conservative: fires only on
    the unambiguous middle SECONDARY endings, and never when a -θη- passive
    marker is present, so it cannot touch a real passive; aorist
    subjunctive/optative middles fall through to the flaggers."""
    if feats.get("Voice") != "Mid,Pass":
        return None
    if not (feats.get("Tense") == "Past" and feats.get("Aspect") == "Perf"):
        return None
    ds = _stem_tail(form)
    if "θη" in ds:
        return None
    if _AOR_MID_END_RE.search(ds):
        return ({"Voice": "Mid"}, "aorist-mid-voice",
                f"{form!r}: aorist middle ending; the aorist keeps Mid distinct "
                f"from the -θη- passive, so Mid not the syncretic Mid,Pass")
    return None


# ---------------------------------------------------------------------------
# Contested flaggers (do NOT auto-apply; they over-fire on convention noise)
# ---------------------------------------------------------------------------
# Each yields (field, suggested_value, evidence, signal_name). The suggested
# value is the deterministic signal an adjudicator must later AGREE WITH to
# promote the field. ``suggested_value`` may be None (we have a suspicion but
# no concrete value).


def _flag_elided_present(form, feats):
    """An elided present whose ending could be an (unaugmented) past.

    This is the demoted elision-tense rule: Homer routinely omits the augment,
    so -τʼ = present -ται OR past -το, and -ντʼ = present -νται OR past -ντο --
    the surface form alone cannot decide (see γίγνετʼ / ἐπαυρίσκοντʼ). So we
    flag the tense contested with a suggested Past for the adjudicator to
    confirm against the context, and never force it; a genuine elided present
    mediopassive must survive."""
    if feats.get("Tense") != "Pres" or not _is_elided(form):
        return
    tail = _stem_tail(form)
    if tail.endswith("ντ"):
        yield ("Tense", "Past",
               f"{form!r}: elided -ντʼ may be a past -ντο (or present mp "
               f"-νται); present is suspect", "elided-present")
    elif tail.endswith("τ"):
        yield ("Tense", "Past",
               f"{form!r}: elided -τʼ may be a past secondary -το (or present "
               f"mp -ται); present is suspect", "elided-present")


def _lexicon_flags(chosen, hints):
    """Lexicon signals (lemmatizer lemma, transitive-sense vs voice)."""
    if not hints:
        return
    ll = hints.get("lemmatizer_lemma")
    if ll and ll != chosen.get("lemma"):
        yield ("lemma", ll,
               f"lemmatizer says {ll!r}, the treebank says "
               f"{chosen.get('lemma')!r}", "lemmatizer-lemma")
    if hints.get("transitive_sense") and \
            chosen["feats"].get("Voice") in _MEDIOPASSIVE:
        yield ("Voice", "Act",
               "the line is filed under a transitive sense but the verb is "
               "tagged non-active", "lexicon-transitive")


def _syntax_flags(chosen, hints):
    """Syntax signals (subject/verb number agreement)."""
    if not hints:
        return
    sn = hints.get("subject_number")
    vn = hints.get("verb_number") or chosen["feats"].get("Number")
    if sn and vn and sn != vn:
        yield ("Number", sn,
               f"the subject is {sn} but the verb is tagged {vn}",
               "syntax-subj-number")
    elif hints.get("subj_verb_number_mismatch"):
        yield ("Number", sn,
               "subject/verb number mismatch in the dependency tree",
               "syntax-subj-number")


# ---------------------------------------------------------------------------
# Adjudication (optional; the AND-gate)
# ---------------------------------------------------------------------------


def _adjudicate(tok, adjudicator, suggestions):
    """Call the injected adjudicator on a contested token and promote a field
    only when the adjudicator's change AND a deterministic signal agree."""
    try:
        verdict = adjudicator(tok.to_dict()) or {}
    except Exception as exc:  # an adjudicator failure must not lose the token
        tok.provenance.append(Provenance("adjudicator:error", str(exc), 0.0))
        return
    changes = verdict.get("changes") or {}
    conf = float(verdict.get("confidence") or 0.0)
    promoted = []
    for fld, val in changes.items():
        sig = suggestions.get(fld)
        if sig and sig[0] is not None and sig[0] == val:
            if fld in ("lemma", "upos"):
                tok.chosen[fld] = val
            else:
                tok.chosen["feats"][fld] = val
            promoted.append(fld)
            tok.provenance.append(Provenance(
                f"adjudicator+{sig[1]}",
                f"adjudicator and {sig[1]} agree on {fld}={val!r}", conf))
        else:
            tok.provenance.append(Provenance(
                "adjudicator:suggest",
                f"adjudicator suggests {fld}={val!r} with no agreeing "
                f"deterministic signal; left contested", conf))
    if promoted:
        tok.contested_fields = [f for f in tok.contested_fields
                                if f not in promoted]
        tok.contested = bool(tok.contested_fields)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconcile_token(form, readings, *, lexicon_hints=None, syntax_hints=None,
                    perf_stems=None, adjudicator=None, meta=None):
    """Reconcile one token's analysis across its source readings.

    Args:
        form: the surface form (NFC polytonic).
        readings: source readings, in priority order. Each is a ``Reading``
            or a ``{"source", "lemma", "upos", "feats"}`` dict.
        lexicon_hints: optional ``{"lemmatizer_lemma": str,
            "transitive_sense": bool}`` -- caller-computed lexicon signals.
        syntax_hints: optional ``{"subject_number": str, "verb_number": str,
            "subj_verb_number_mismatch": bool}`` -- caller-computed syntax
            signals.
        perf_stems: override / extend the perfect-stem table (defaults to
            :data:`PERF_STEMS`).
        adjudicator: optional callable ``token_dict -> {"changes": {...},
            "confidence": float}``; invoked only on contested tokens, and a
            change is applied only if a deterministic signal agrees.
        meta: positional/identity passthrough (line, col, idx, ...).

    Returns:
        A :class:`ReconciledToken`.
    """
    rs = _coerce_readings(readings)
    chosen, disagree = vote_readings(rs)
    feats = chosen["feats"]
    provenance = [Provenance(
        "vote",
        f"{len(rs)} source(s); "
        f"disagree on {', '.join(disagree) if disagree else 'nothing'}",
        1.0 if not disagree else 0.5)]

    contested = set(disagree)        # cross-source disagreement (a flagger)
    resolved = set()                 # fields a high-precision rule decided
    # suggestions[field] = (deterministic value, signal name) the adjudicator
    # must match to promote. Cross-source-contested fields default to the
    # vote majority as their signal.
    suggestions = {}
    for f in disagree:
        win, tie = _majority(_field_values(rs, f))
        if win is not None and not tie:
            suggestions[f] = (win, "vote")

    if _is_verbal(chosen):
        for res in (
            rule_mid_ending(form, feats),
            rule_perf_stem(form, chosen.get("lemma"), feats,
                           PERF_STEMS if perf_stems is None else perf_stems),
            rule_aorist_mid_voice(form, feats),
        ):
            if not res:
                continue
            changes, name, evidence = res
            feats.update(changes)
            for f in changes:
                resolved.add(f)
                contested.discard(f)     # forced -> no longer contested
                suggestions.pop(f, None)
            provenance.append(Provenance(f"rule:{name}", evidence, 1.0))
        for f, val, ev, sig in _flag_elided_present(form, feats):
            if f not in resolved:
                contested.add(f)
                provenance.append(Provenance(f"flag:{sig}", ev, 0.4))
                suggestions.setdefault(f, (val, sig))

    for flagger in (_lexicon_flags(chosen, lexicon_hints),
                    _syntax_flags(chosen, syntax_hints)):
        for f, val, ev, sig in flagger:
            if f in resolved:
                continue
            contested.add(f)
            provenance.append(Provenance(f"flag:{sig}", ev, 0.4))
            suggestions[f] = (val, sig)      # an explicit signal wins the slot

    tok = ReconciledToken(
        form=_nfc(form), chosen=chosen, readings=rs,
        contested=bool(contested), contested_fields=sorted(contested),
        provenance=provenance, meta=dict(meta or {}))

    if adjudicator is not None and tok.contested:
        _adjudicate(tok, adjudicator, suggestions)
    return tok


def reconcile_work(tokens, *, perf_stems=None, adjudicator=None):
    """Reconcile an iterable of token dicts.

    Each item is ``{"form", "readings", "lexicon_hints"?, "syntax_hints"?,
    "meta"?}``. Returns a list of :class:`ReconciledToken`."""
    return [
        reconcile_token(
            t["form"], t.get("readings") or [],
            lexicon_hints=t.get("lexicon_hints"),
            syntax_hints=t.get("syntax_hints"),
            perf_stems=perf_stems, adjudicator=adjudicator,
            meta=t.get("meta"))
        for t in tokens
    ]
