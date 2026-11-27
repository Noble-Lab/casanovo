import depthcharge.tokenizers.peptides
import numpy as np
import pytest

from casanovo.denovo.evaluate import (
    aa_match,
    aa_match_batch,
    aa_match_metrics,
    precision_coverage,
)


def test_eval_metrics():
    """
    Test peptide and amino acid-level evaluation metrics.
    Predicted AAs are considered correct if they are <0.1Da from the
    corresponding ground truth AA with either a suffix or prefix <0.5Da from
    the ground truth. A peptide prediction is correct if all its AA are correct
    matches.
    """
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer()

    preds = [
        "SPEIK",
        "SPAEL",
        "SPAEKL",
        "ASPEKL",
        "SPEKL",
        "SPELQ",
        "PSEKL",
        "SPEK",
    ]
    gt = len(preds) * ["SPELK"]

    aa_matches, n_pred_aa, n_gt_aa = aa_match_batch(
        peptides1=preds,
        peptides2=gt,
        aa_dict=tokenizer.residues,
        mode="best",
    )

    assert n_pred_aa == 41
    assert n_gt_aa == 40

    aa_precision, aa_recall, pep_precision = aa_match_metrics(
        aa_matches, n_gt_aa, n_pred_aa
    )

    assert 2 / 8 == pytest.approx(pep_precision)
    assert 26 / 40 == pytest.approx(aa_recall)
    assert 26 / 41 == pytest.approx(aa_precision)

    aa_matches, pep_match = aa_match(None, None, tokenizer.residues)

    assert aa_matches.shape == (0,)
    assert not pep_match

    aa_matches, pep_match = aa_match("PEPTIDE", None, tokenizer.residues)

    assert np.array_equal(aa_matches, np.zeros(len("PEPTIDE"), dtype=bool))
    assert not pep_match


def test_precision_coverage():
    scores = [0.6, 0.9, 0.7, 0.8]
    pep_matches = [False, True, True, False]
    order, coverage, precision = precision_coverage(
        scores, pep_matches, n_spectra=5
    )
    # Ranked by decreasing score (0.9, 0.8, 0.7, 0.6) -> indices 1, 3, 2, 0.
    assert list(order) == [1, 3, 2, 0]
    # Matches in that order (True, False, True, False) -> cumsum 1, 1, 2, 2.
    np.testing.assert_allclose(precision, [1.0, 0.5, 2 / 3, 0.5])
    np.testing.assert_allclose(coverage, [0.2, 0.4, 0.6, 0.8])

    # Tied scores keep their original order (stable sort).
    order, coverage, precision = precision_coverage(
        [0.5, 0.5, 0.9], [True, False, True], n_spectra=3
    )
    assert list(order) == [2, 0, 1]
    np.testing.assert_allclose(precision, [1.0, 1.0, 2 / 3])
