"""Pinned HuggingFace Hub revision for reproducible tagger installs.

The tagger pins the exact commit SHA of the weights it downloads. This means:

- a tagger install today pulls the same blobs as the same version in a
  year, even if the upstream HF repo has been updated.
- Security: a compromised HF account can't silently swap weights under us.
- Reproducibility: benchmark numbers stay meaningful across time.

To cut a new release that tracks updated weights:
  1. Fetch the latest SHA (HF UI or `huggingface_hub.list_repo_commits`).
  2. Update the constant below.
  3. Re-run the slow test suite to confirm behavior.
  4. Bump dilemma.tagger's __version__.
"""

# Our own trained tagger weights, shipped under ciscoriordan/dilemma at
# tagger/<lang>/{tagger.onnx, tagger_labels.json, tokenizer/, mwt.json}. Pinned
# to the commit that ships the grc + el biaffine dependency-parse models.
TAGGER_WEIGHTS_REV = "15a62610727b32253b780d2492aa1b546ac66443"
