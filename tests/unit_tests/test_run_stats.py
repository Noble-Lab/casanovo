import pandas as pd
import numpy as np
import random
import string

from casanovo.utils import (
    get_score_bins,
    get_peptide_lengths,
    get_peptide_length_histo,
)

np.random.seed(4000)
random.seed(4000)


def test_get_score_bins():
    NUM_TEST = 5
    NUM_BINS = 5
    BIN_MIN = -1.0
    BIN_MAX = 1.0
    BIN_RNG = BIN_MAX - BIN_MIN
    MAX_NUM = 10
    MIN_NUM = 1

    for _ in range(NUM_TEST):
        curr_bins = (np.random.rand(NUM_BINS) * BIN_RNG) + BIN_MIN
        curr_bins = np.sort(curr_bins)
        nums_per_bin = np.random.randint(MIN_NUM, MAX_NUM, NUM_BINS)
        expected = dict()
        curr_scores = np.array([])
        cumulative_sum = 0

        for i in range(len(nums_per_bin) - 1, -2, -1):
            curr_min = BIN_MIN if i < 0 else curr_bins[i]
            curr_max = (
                BIN_MAX if i + 1 >= len(nums_per_bin) else curr_bins[i + 1]
            )
            curr_num = nums_per_bin[i]
            next_scores = (
                np.random.rand(curr_num) * (curr_max - curr_min)
            ) + curr_min
            curr_scores = np.append(curr_scores, next_scores)
            cumulative_sum += curr_num

            if i >= 0:
                expected[curr_min] = cumulative_sum

        np.random.shuffle(curr_scores)
        results_table = pd.DataFrame({"score": curr_scores})
        actual = get_score_bins(results_table, curr_bins)
        assert expected == actual


def test_get_peptide_lengths():
    NUM_TEST = 5
    MAX_LENGTH = 20
    MIN_LENGTH = 5
    MAX_NUM_PEPTIDES = 200
    MIN_NUM_PEPTIDES = 20
    PROB_MASS_MOD = 0.1

    num_peptides = np.random.randint(
        MIN_NUM_PEPTIDES, MAX_NUM_PEPTIDES, NUM_TEST
    )
    for curr_num_peptides in num_peptides:
        expected = np.random.randint(MIN_LENGTH, MAX_LENGTH, curr_num_peptides)
        peptide_list = []

        for curr_expected_len in expected:
            curr_peptide_seq = ""

            i = 0
            while i < curr_expected_len:
                if random.random() < PROB_MASS_MOD:
                    random_mass_mod = 50 * random.random()
                    random_mass_mod = (
                        f"{random.choice('+-')}{random_mass_mod:.5f}"
                    )
                    curr_peptide_seq += random_mass_mod
                    continue

                random_peptide = random.choice(string.ascii_uppercase)
                curr_peptide_seq += random_peptide
                i += 1

            peptide_list.append(curr_peptide_seq)

        results_table = pd.DataFrame({"sequence": peptide_list})
        actual = get_peptide_lengths(results_table)
        assert np.array_equal(expected, actual)


def test_get_peptide_length_histo():
    NUM_TEST = 5
    MAX_LENGTH = 20
    MIN_LENGTH = 1
    MAX_FREQUENCY = 100
    MIN_FREQUENCY = 10
    MAX_NUM_SAMPLES = MAX_LENGTH - MIN_LENGTH + 1

    for _ in range(NUM_TEST):
        num_samples = random.randint(1, MAX_NUM_SAMPLES)
        lengths = random.sample(
            range(MIN_LENGTH, MAX_LENGTH + 1, 1), k=num_samples
        )
        frequencies = np.random.randint(
            MIN_FREQUENCY, MAX_FREQUENCY, len(lengths)
        )

        nums_list = list()
        for curr_length, curr_frequency in zip(lengths, frequencies.tolist()):
            nums_list += [curr_length] * curr_frequency

        peptide_lengths = np.array(nums_list)
        np.random.shuffle(peptide_lengths)
        expected = dict(zip(lengths, frequencies.tolist()))
        actual = get_peptide_length_histo(peptide_lengths)
        assert expected == actual
