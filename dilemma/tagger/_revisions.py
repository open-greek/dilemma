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
# to the commit that ships the grc tagger fine-tuned on the Iliad composite
# gold + a GLAUx mixture (GLAUx test strict 93.6->94.1, UAS 83.6->85.1;
# held-out Iliad books 6/22 engine-only strict 82->87.6/88.4). Also carries
# the grc training checkpoint (tagger_model.pt) for future warm-starts.
TAGGER_WEIGHTS_REV = "876efd48e1498c354f43742c1ba546b311006f2f"
