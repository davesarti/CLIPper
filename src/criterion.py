"""The assignment's ground-truth rule, in one place.

Assignment S3.1.1: a retrieved image is correct **iff** (1) it satisfies the
query's positive/negative constraints, and (2) its remaining attributes are
within Hamming distance 2 of the reference's. Verified against
celeba_evaluation.json by exact set reconstruction on all 33,052
(query, reference) pairs.

Both halves are statements about 40-bit attribute codes, so both are computed
here from labels alone - no embeddings, no model. Everything that needs the rule
imports it from this module: the val benchmark (src/evaluation.py), the miner
(src/mining.py) and the in-batch false-negative mask (src/training.py). The rule
used to live inline inside build_val_benchmark while the miner carried a
different one, which is precisely the drift this module exists to prevent
(docs/method-proposal-mining-alignment.md S1).

The two conditions also partition the wrong answers into the two families the
miner uses as negatives:

    valid     =  satisfies & (hamming <= 2)     a correct answer
    violators = ~satisfies & (hamming <= 2)     right person, broken constraint
    drifters  =  satisfies & (hamming >  2)     right constraints, wrong person
"""

import torch

MAX_HAMMING = 2   # assignment S3.1.1 (2)
MIN_TARGETS = 5   # assignment S3.1.1 inclusion rule: ">= 5 valid ground-truth targets"


def satisfies(labels: torch.Tensor, add: list[int], remove: list[int]) -> torch.Tensor:
    """(N,) bool: has every attribute in `add` and none in `remove`.

    labels: (N, A) bool. An empty query is satisfied by everything.
    """
    ok = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
    if add:
        ok &= labels[:, add].all(dim=1)
    if remove:
        ok &= ~labels[:, remove].any(dim=1)
    return ok


def hamming_to(
    labels: torch.Tensor,
    code: torch.Tensor,
    queried: list[int],
) -> torch.Tensor:
    """(N,) int: disagreements with `code` over the NON-queried attributes.

    labels: (N, A) bool; code: (A,) bool; queried: attribute rows the query
    names. Those differ from the reference by construction - that is what the
    query asked for - so S3.1.1 excludes them from the distance.

    Computed as "all disagreements minus the queried ones" rather than by
    slicing the complement: it avoids materialising a second (N, A-|queried|)
    copy on every call, and the miner makes one call per sampled query.
    """
    diff = labels != code
    total = diff.sum(dim=1)
    if not queried:
        return total
    return total - diff[:, queried].sum(dim=1)


def valid_mask(
    labels: torch.Tensor,
    code: torch.Tensor,
    add: list[int],
    remove: list[int],
    max_hamming: int = MAX_HAMMING,
) -> torch.Tensor:
    """(N,) bool: the S3.1.1 correct answers for this reference code and query.

    Does not exclude the reference itself - callers that hold its row index are
    the ones that can, and the batch-level mask in src/training.py works on a
    code without knowing any index.
    """
    queried = list(add) + list(remove)
    return satisfies(labels, add, remove) & (
        hamming_to(labels, code, queried) <= max_hamming
    )
