import depthcharge.tokenizers
import numpy as np
import pandas as pd
import pytest

from casanovo.data import db_utils


def test_export(tiny_fasta_file, tmp_path):
    """Test for export feature in db search"""
    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="partial",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=100,
        max_mods=0,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )

    pdb.export(tmp_path, "output")
    output_file = tmp_path / "output.tsv"
    ground_truth = pdb.db_peptides

    loaded = pd.read_csv(output_file, sep="\t", index_col=0)
    ground_truth["protein"] = ground_truth["protein"].apply(str)
    loaded["protein"] = loaded["protein"].astype(str)

    pd.testing.assert_frame_equal(ground_truth, loaded)


@pytest.mark.parametrize(
    "enzyme, specificity, expected, use_sort",
    [
        (
            "N/A",
            "non-specific",
            [
                ("", "foo"),
                ("M", "foo"),
                ("ME", "foo"),
                ("", "foo"),
                ("E", "foo"),
                ("", "corrupted"),
                ("M", "corrupted"),
                ("ME", "corrupted"),
                ("", "corrupted"),
                ("E", "corrupted"),
                ("", "corrupted"),
            ],
            False,
        ),
        (
            "trypsin",
            "partial",
            [
                ("M", "foo"),
                ("ME", "foo"),
                ("E", "foo"),
                ("ME", "corrupted"),
                ("M", "corrupted"),
            ],
            True,
        ),
        (
            ".Q.",
            "partial",
            [
                ("M", "foo"),
                ("ME", "foo"),
                ("E", "foo"),
                ("ME", "corrupted"),
                ("M", "corrupted"),
            ],
            True,
        ),
        (
            "trypsin",
            "full",
            [("ME", "foo")],
            True,
        ),
        (
            ".Q.",
            "full",
            [("ME", "foo")],
            True,
        ),
    ],
)
def test_digestion_with_unknown_amino_acids(
    enzyme, specificity, expected, use_sort, tmp_path
):
    """Test digestion with unknown amino acids"""
    fasta_path = tmp_path / "tiny_fasta.fasta"
    fasta_path.write_text(">foo\nME\n>corrupted\nMEX\n", encoding="utf-8")

    valid_aa = list("ARNDCEQGHILKMFPSTWYV")
    min_len = 0
    max_len = 18

    results = list(
        db_utils._peptide_generator(
            str(fasta_path),
            enzyme,
            specificity,
            0,
            min_len,
            max_len,
            valid_aa,
        )
    )

    result_peptides = [p for p, _ in results]
    expected_peptides = [p for p, _ in expected]

    if use_sort:
        assert sorted(result_peptides) == sorted(
            expected_peptides
        ), f"Failed for enzyme={enzyme}, specificity={specificity}"
    else:
        assert (
            result_peptides == expected_peptides
        ), f"Failed for enzyme={enzyme}, specificity={specificity}"


def test_peptide_generator_errors(tiny_fasta_file):
    residues_dict = (
        depthcharge.tokenizers.PeptideTokenizer.from_massivekb().residues
    )
    with pytest.raises(FileNotFoundError):
        [
            (a, b)
            for a, b in db_utils._peptide_generator(
                "fail.fasta", "trypsin", "full", 0, 5, 10, residues_dict
            )
        ]
    with pytest.raises(ValueError):
        [
            (a, b)
            for a, b in db_utils._peptide_generator(
                tiny_fasta_file, "trypsin", "fail", 0, 5, 10, residues_dict
            )
        ]


def test_to_neutral_mass():
    mz = 500
    charge = 2
    neutral_mass = db_utils._to_neutral_mass(mz, charge)
    assert neutral_mass == 997.98544706646

    mz = 500
    charge = 1
    neutral_mass = db_utils._to_neutral_mass(mz, charge)
    assert neutral_mass == 498.99272353323


def test_peptide_generator_unknown_amino_acids(tmp_path):
    valid_aa = set("ARNDCEQGHILKMFPSTWYV")
    fasta_path = tmp_path / "tiny.fasta"
    fasta_path.write_text(">foo\nME\n>corrupted\nMEX\n", encoding="utf-8")

    non_specific = list(
        db_utils._peptide_generator(
            str(fasta_path), "N/A", "non-specific", 0, 1, 10, valid_aa
        )
    )
    assert sorted(non_specific) == sorted(
        [
            ("M", "foo"),
            ("ME", "foo"),
            ("E", "foo"),
            ("M", "corrupted"),
            ("ME", "corrupted"),
            ("E", "corrupted"),
        ]
    )

    enzymatic = list(
        db_utils._peptide_generator(
            str(fasta_path), "trypsin", "full", 0, 1, 10, valid_aa
        )
    )
    assert enzymatic == [("ME", "foo")]


def test_digest_fasta_cleave(tiny_fasta_file):
    # No missed cleavages
    expected_normal = [
        "ATSIPAR",
        "VTLSC[Carbamidomethyl]R",
        "LLIYGASTR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
    ]

    # 1 missed cleavage
    expected_1missedcleavage = [
        "ATSIPAR",
        "VTLSC[Carbamidomethyl]R",
        "LLIYGASTR",
        "LLIYGASTRATSIPAR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "EIVMTQSPPTLSLSPGERVTLSC[Carbamidomethyl]R",
        "VTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQQKPGQAPRLLIYGASTR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
        "MEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGER",
        "ATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
    ]

    # 3 missed cleavages
    expected_3missedcleavage = [
        "ATSIPAR",
        "VTLSC[Carbamidomethyl]R",
        "LLIYGASTR",
        "LLIYGASTRATSIPAR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "EIVMTQSPPTLSLSPGERVTLSC[Carbamidomethyl]R",
        "VTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQQKPGQAPRLLIYGASTR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
        "ASQSVSSSYLTWYQQKPGQAPRLLIYGASTRATSIPAR",
        "VTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPRLLIYGASTR",
        "MEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGER",
        "ATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
        "VTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPRLLIYGASTRATSIPAR",
        "MEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGERVTLSC[Carbamidomethyl]R",
        "EIVMTQSPPTLSLSPGERVTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPR",
        "LLIYGASTRATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
    ]
    for missed_cleavages, expected in zip(
        (0, 1, 3),
        (expected_normal, expected_1missedcleavage, expected_3missedcleavage),
    ):
        pdb = db_utils.ProteinDatabase(
            fasta_path=str(tiny_fasta_file),
            enzyme="trypsin",
            digestion="full",
            missed_cleavages=missed_cleavages,
            min_peptide_len=6,
            max_peptide_len=50,
            max_mods=0,
            precursor_tolerance=20,
            isotope_error=[0, 0],
            allowed_fixed_mods="C:C[Carbamidomethyl]",
            allowed_var_mods=(
                "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
                "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
            ),
            tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
        )
        assert pdb.db_peptides.index.to_list() == expected


def test_digest_fasta_mods(tiny_fasta_file):
    # 1 modification allowed
    # fixed: C+57.02146
    # variable: 1M+15.994915,1N+0.984016,1Q+0.984016
    # nterm: 1X+42.010565,1X+43.005814,1X-17.026549,1X+25.979265
    expected = np.array(
        [
            697.37586,  # -17.027ATSIPAR
            714.40241,  # ATSIPAR
            717.34797,  # -17.027VTLSC+57.021R
            734.37452,  # VTLSC+57.021R
            740.3817,  # +43.006-17.027ATSIPAR
            756.41297,  # +42.011ATSIPAR
            757.40821,  # +43.006ATSIPAR
            760.35374,  # +43.006-17.027VTLSC+57.021R
            776.38502,  # +42.011VTLSC+57.021R
            777.38032,  # +43.006VTLSC+57.021R
            975.53889,  # -17.027LLIYGASTR
            992.5655,  # LLIYGASTR
            1018.54472,  # +43.006-17.027LLIYGASTR
            1034.57599,  # +42.011LLIYGASTR
            1035.57123,  # +43.006LLIYGASTR
            1923.96125,  # -17.027EIVMTQSPPTLSLSPGER
            1940.98774,  # EIVMTQSPPTLSLSPGER
            1941.97175,  # EIVMTQ+0.984SPPTLSLSPGER
            1956.98261,  # EIVM+15.995TQSPPTLSLSPGER
            1966.96708,  # +43.006-17.027EIVMTQSPPTLSLSPGER
            1982.99836,  # +42.011EIVMTQSPPTLSLSPGER
            1983.99372,  # +43.006EIVMTQSPPTLSLSPGER
            2323.26496,  # -17.027MEAPAQLLFLLLLWLPDTTR
            2339.25983,  # -17.027M+15.995EAPAQLLFLLLLWLPDTTR
            2340.29157,  # MEAPAQLLFLLLLWLPDTTR
            2341.27546,  # MEAPAQ+0.984LLFLLLLWLPDTTR
            2356.28644,  # M+15.995EAPAQLLFLLLLWLPDTTR
            2366.2708,  # +43.006-17.027MEAPAQLLFLLLLWLPDTTR
            2382.26567,  # +43.006-17.027M+15.995EAPAQLLFLLLLWLPDTTR
            2382.30207,  # +42.011MEAPAQLLFLLLLWLPDTTR
            2383.29743,  # +43.006MEAPAQLLFLLLLWLPDTTR
            2398.29694,  # +42.011M+15.995EAPAQLLFLLLLWLPDTTR
            2399.2923,  # +43.006M+15.995EAPAQLLFLLLLWLPDTTR
            2451.18171,  # -17.027ASQSVSSSYLTWYQQKPGQAPR
            2468.20832,  # ASQSVSSSYLTWYQQKPGQAPR
            2469.19221,  # ASQSVSSSYLTWYQ+0.984QKPGQAPR
            2469.19245,  # ASQSVSSSYLTWYQQ+0.984KPGQAPR
            2469.19245,  # ASQ+0.984SVSSSYLTWYQQKPGQAPR
            2469.19245,  # ASQSVSSSYLTWYQQKPGQ+0.984APR
            2494.18779,  # +43.006-17.027ASQSVSSSYLTWYQQKPGQAPR
            2510.21906,  # +42.011ASQSVSSSYLTWYQQKPGQAPR
            2511.21418,  # +43.006ASQSVSSSYLTWYQQKPGQAPR
            3789.66681,  # -17.027FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
            3806.69318,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
            3807.67707,  # FSGSGSGTDFTLTISSLQ+0.984PEDFAVYYC+57.021QQDYNLP
            3807.67707,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYN+0.984LP
            3807.67731,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021Q+0.984QDYNLP
            3807.67731,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQ+0.984DYNLP
            3832.67265,  # +43.006-17.027FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
            3848.70392,  # +42.011FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
            3849.69904,  # +43.006FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
        ]
    )

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=1,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )

    assert np.allclose(pdb.db_peptides["calc_mass"].to_numpy(), expected)


def test_length_restrictions(tiny_fasta_file):
    # length between 20 and 50
    expected_long = [
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
    ]

    # length between 6 and 8
    expected_short = ["ATSIPAR", "VTLSC[Carbamidomethyl]R"]

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=20,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_long

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=8,
        max_mods=0,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_short


@pytest.mark.parametrize(
    "enzyme,digestion,expected",
    [
        pytest.param(
            "arg-c",
            "full",
            np.array(
                [
                    714.40241,  # ATSIPAR
                    734.37452,  # VTLSC+57.021R
                    992.5655,  # LLIYGASTR
                    1940.98774,  # EIVMTQSPPTLSLSPGER
                    2340.29157,  # MEAPAQLLFLLLLWLPDTTR
                    2468.20832,  # ASQSVSSSYLTWYQQKPGQAPR
                    3806.69318,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
                ]
            ),
            id="arg-c",
        ),
        pytest.param(
            "asp-n",
            "full",
            np.array(
                [
                    1192.48591,  # DFAVYYC+57.021QQ
                    1349.67145,  # DFTLTISSLQPE
                    1867.06818,  # MEAPAQLLFLLLLWLP
                ]
            ),
            id="asp-n",
        ),
        pytest.param(
            "R",
            "full",
            np.array(
                [
                    714.40241,  # ATSIPAR
                    734.37452,  # VTLSC+57.021R
                    992.5655,  # LLIYGASTR
                    1940.98774,  # EIVMTQSPPTLSLSPGER
                    2340.29157,  # MEAPAQLLFLLLLWLPDTTR
                    2468.20832,  # ASQSVSSSYLTWYQQKPGQAPR
                    3806.69318,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
                ]
            ),
            id="regex-R",
        ),
        pytest.param(
            "trypsin",
            "partial",
            np.array(
                [
                    540.21796,  # FSGSGS
                    558.30128,  # ATSIPA
                    577.27076,  # ASQSVS
                    624.33436,  # PGQAPR
                    643.3653,  # TSIPAR
                    645.27924,  # MEAPAQ
                    648.38459,  # LLIYGA
                    653.3133,  # YGASTR
                    657.34455,  # LSPGER
                    701.37079,  # LPDTTR
                    719.35236,  # EIVMTQ
                    734.37452,  # VTLSC+57.021R
                    748.33912,  # QDYNLP
                ]
            ),
            id="semispecific",
        ),
        pytest.param(
            "trypsin",
            "non-specific",
            np.array(
                [
                    450.17106,  # SGSGSG
                    464.18665,  # GSGSGT
                    522.19211,  # SGSGTD
                    540.21796,  # FSGSGS
                    558.30128,  # ATSIPA
                    561.28705,  # GASTRA
                    572.31696,  # LSLSPG
                    577.27076,  # ASQSVS
                    582.22858,  # GSGTDF
                    588.27546,  # SLSPGE
                    593.26569,  # QSVSSS
                    593.26569,  # SQSVSS
                    596.32819,  # KPGQAP
                    600.3119,  # SPPTLS
                    605.3133,  # ASTRAT
                    609.28711,  # RFSGSG
                    610.29621,  # IYGAST
                    611.36426,  # APAQLL
                    616.34321,  # PTLSLS
                    616.34321,  # TLSLSP
                    620.33808,  # TLTISS
                    621.30817,  # STRATS
                    622.33259,  # LIYGAS
                    623.30274,  # ARFSGS
                    624.33436,  # PGQAPR
                    626.25477,  # SGTDFT
                    626.3639,  # PPTLSL
                    627.32282,  # EAPAQL
                    627.33405,  # QKPGQA
                    628.27045,  # SVSSSY
                    629.30207,  # TQSPPT
                    632.37446,  # LTISSL
                    633.32343,  # PARFSG
                    640.36567,  # GQAPRL
                    641.33845,  # QSPPTL
                    643.32892,  # SPGERV
                    643.35407,  # ISSLQP
                    643.3653,  # TSIPAR
                    643.3653,  # RATSIP
                    645.27924,  # MEAPAQ
                    646.33985,  # RASQSV
                    647.349,  # TISSLQ
                    647.36024,  # TRATSI
                    648.38459,  # LLIYGA
                    652.30683,  # GTDFTL
                    653.3133,  # YGASTR
                    654.32252,  # VSSSYL
                    656.3017,  # SSSYLT
                    657.34455,  # LSPGER
                    657.34455,  # PGERVT
                    659.29487,  # MTQSPP
                    659.31263,  # SSLQPE
                    661.31049,  # VMTQSP
                    673.37586,  # GERVTL
                    676.30683,  # PEDFAV
                    677.3418,  # IVMTQS
                    680.37446,  # FTLTIS
                    681.45374,  # APRLLI
                    684.35554,  # QQKPGQ
                    687.30756,  # SLQPED
                    687.39551,  # PAQLLF
                    689.38605,  # IPARFS
                    689.38605,  # SIPARF
                    692.32758,  # LSC+57.021RAS
                    696.33301,  # TDFTLT
                    696.42817,  # QAPRLL
                    701.37079,  # LPDTTR
                    703.38642,  # ERVTLS
                    703.42682,  # AQLLFL
                    705.29694,  # QPEDFA
                    706.34321,  # TLSC+57.021RA
                    707.30207,  # SC+57.021RASQ
                    707.30213,  # C+57.021RASQS
                    708.36939,  # DFTLTI
                    717.32929,  # PDTTRE
                    717.40204,  # TTREIV
                    719.35236,  # EIVMTQ
                    719.3603,  # YQQKPG
                    730.49927,  # LFLLLL
                    730.49927,  # LLFLLL
                    731.349,  # WLPDTT
                    733.36066,  # DTTREI
                    733.44861,  # RLLIYG
                    734.37452,  # RVTLSC+57.021
                    734.37452,  # VTLSC+57.021R
                    742.31733,  # EDFAVY
                    743.38544,  # LWLPDT
                    745.47376,  # QLLFLL
                    747.34394,  # LQPEDF
                    747.39484,  # TREIVM
                    747.3949,  # REIVMT
                    748.33912,  # QDYNLP
                    753.47883,  # LLLWLP
                    755.349,  # SSYLTW
                    755.42176,  # LLWLPD
                    769.51014,  # LLLLWL
                    773.47993,  # PRLLIY
                    776.33814,  # DFAVYY
                    779.34492,  # QQDYNL
                    802.33198,  # AVYYC+57.021Q
                    803.49451,  # FLLLLW
                    821.34186,  # FAVYYC+57.021
                    826.29157,  # C+57.021QQDYN
                    831.38032,  # SYLTWY
                    837.40217,  # LTWYQQ
                    848.41816,  # WYQQKP
                    852.41303,  # TWYQQK
                    859.35346,  # VYYC+57.021QQ
                    872.4068,  # YLTWYQ
                    875.31196,  # YYC+57.021QQD
                    875.31202,  # YC+57.021QQDY
                ]
            ),
            id="nonspecific",
        ),
    ],
)
def test_digest_fasta_enzyme(tiny_fasta_file, enzyme, digestion, expected):
    """Test peptide digestion behavior for various enzymes and digest modes."""
    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme=enzyme,
        digestion=digestion,
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=50 if digestion == "full" else 6,
        max_mods=0,
        precursor_tolerance=20 if digestion == "full" else 10000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )

    assert np.allclose(pdb.db_peptides["calc_mass"].to_numpy(), expected)


def test_get_candidates(tiny_fasta_file):
    # precursor_window is 10000
    expected_smallwindow = ["LLIYGASTR"]

    # precursor window is 150000
    expected_midwindow = ["LLIYGASTR"]

    # precursor window is 600000
    expected_widewindow = ["ATSIPAR", "VTLSC[Carbamidomethyl]R", "LLIYGASTR"]

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=1,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_smallwindow == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=1,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=150000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_midwindow == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=1,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=600000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_widewindow == list(candidates)


def test_get_candidates_isotope_error(tiny_fasta_file):
    # Tide isotope error windows for 496.2, 2+:
    # 0: [980.481617, 1000.289326]
    # 1: [979.491114, 999.278813]
    # 2: [978.500611, 998.268300]
    # 3: [977.510108, 997.257787]

    peptide_list = [
        ("A", 1001, "foo"),
        ("B", 1000, "foo"),
        ("C", 999, "foo"),
        ("D", 998, "foo"),
        ("E", 997, "foo"),
        ("F", 996, "foo"),
        ("G", 995, "foo"),
        ("H", 994, "foo"),
        ("I", 993, "foo"),
        ("J", 992, "foo"),
        ("K", 991, "foo"),
        ("L", 990, "foo"),
        ("M", 989, "foo"),
        ("N", 988, "foo"),
        ("O", 987, "foo"),
        ("P", 986, "foo"),
        ("Q", 985, "foo"),
        ("R", 984, "foo"),
        ("S", 983, "foo"),
        ("T", 982, "foo"),
        ("U", 981, "foo"),
        ("V", 980, "foo"),
        ("W", 979, "foo"),
        ("X", 978, "foo"),
        ("Y", 977, "foo"),
        ("Z", 976, "foo"),
    ]

    peptide_list = pd.DataFrame(
        peptide_list, columns=["peptide", "calc_mass", "protein"]
    ).set_index("peptide")
    peptide_list.sort_values("calc_mass", inplace=True)

    expected_isotope0 = list("UTSRQPONMLKJIHGFEDCB")
    expected_isotope01 = list("VUTSRQPONMLKJIHGFEDCB")
    expected_isotope012 = list("WVUTSRQPONMLKJIHGFEDCB")
    expected_isotope0123 = list("XWVUTSRQPONMLKJIHGFEDCB")

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=0,
        max_peptide_len=0,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    pdb.db_peptides = peptide_list
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope0 == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=0,
        max_peptide_len=0,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 1],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    pdb.db_peptides = peptide_list
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope01 == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=0,
        max_peptide_len=0,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 2],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    pdb.db_peptides = peptide_list
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope012 == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=0,
        max_peptide_len=0,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 3],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    pdb.db_peptides = peptide_list
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope0123 == list(candidates)


def test_ptm_warning(tiny_fasta_file, caplog):
    """Test that a warning is issued for unconfigured modified residues"""
    import logging

    # Use a tokenizer with modified residues
    with caplog.at_level(logging.WARNING):
        db_utils.ProteinDatabase(
            fasta_path=str(tiny_fasta_file),
            enzyme="trypsin",
            digestion="full",
            missed_cleavages=0,
            min_peptide_len=6,
            max_peptide_len=50,
            max_mods=0,
            precursor_tolerance=20,
            isotope_error=[0, 0],
            allowed_fixed_mods="C:C[Carbamidomethyl]",
            allowed_var_mods="nterm:[Acetyl]-",
            tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
        )
    assert any(
        "is not specified as a fixed or variable modification"
        in record.message
        for record in caplog.records
    )
