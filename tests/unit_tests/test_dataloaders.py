import logging
import shutil
import unittest.mock

import lance
import pyarrow as pa
import pytest
import torch

from casanovo.denovo.dataloaders import (
    DeNovoDataModule,
    DistributedBatchSampler,
)


def test_spectrum_id_mgf(mgf_small, tmp_path):
    """Test that spectra from MGF files are specified by their index."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)
    data_module = DeNovoDataModule(
        lance_dir=str(tmp_path),
        train_paths=[mgf_small, mgf_small2],
        valid_paths=[mgf_small, mgf_small2],
        test_paths=[mgf_small, mgf_small2],
        min_peaks=0,
        shuffle=False,
    )
    data_module.setup()

    # train/test combine both files into one dataset (4 spectra total).
    for dataset in [data_module.train_dataset, data_module.test_dataset]:
        for i, (filename, scan_id) in enumerate(
            [
                (mgf_small, "0"),
                (mgf_small, "1"),
                (mgf_small2, "0"),
                (mgf_small2, "1"),
            ]
        ):
            assert dataset[i]["peak_file"][0] == filename.name
            assert dataset[i]["scan_id"][0] == f"index={scan_id}"

    # Validation is now per-file; each dataset has 2 spectra from its file.
    assert len(data_module.valid_datasets) == 2
    for i, filename in enumerate([mgf_small, mgf_small2]):
        vds = data_module.valid_datasets[i]
        assert vds[0]["peak_file"][0] == filename.name
        assert vds[0]["scan_id"][0] == "index=0"
        assert vds[1]["peak_file"][0] == filename.name
        assert vds[1]["scan_id"][0] == "index=1"


def test_log_training_set_size(mgf_small, tmp_path, caplog):
    """Test that the number of training and validation spectra is logged."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)
    data_module = DeNovoDataModule(
        lance_dir=str(tmp_path),
        train_paths=[mgf_small, mgf_small2],
        valid_paths=[mgf_small],
        min_peaks=0,
        shuffle=False,
    )
    with caplog.at_level(logging.INFO, logger="casanovo"):
        data_module.setup()

    assert any(
        msg == "Training dataset contains 4 spectra."
        for msg in caplog.messages
    )
    assert any(
        msg == "Validation dataset contains 2 spectra."
        for msg in caplog.messages
    )


def test_log_training_set_size_shuffled(mgf_small, tmp_path, caplog):
    """Test spectra count logging when shuffle=True (ShufflerIterDataPipe)."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)
    data_module = DeNovoDataModule(
        lance_dir=str(tmp_path),
        train_paths=[mgf_small, mgf_small2],
        valid_paths=[mgf_small],
        min_peaks=0,
        shuffle=True,
    )
    with caplog.at_level(logging.INFO, logger="casanovo"):
        data_module.setup()

    assert any(
        msg == "Training dataset contains 4 spectra."
        for msg in caplog.messages
    )
    assert any(
        msg == "Validation dataset contains 2 spectra."
        for msg in caplog.messages
    )


def _mock_dist(rank, world_size):
    """Pretend to be one rank of an initialized DDP process group."""
    return unittest.mock.patch.multiple(
        "casanovo.denovo.dataloaders.dist",
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_rank=lambda: rank,
        get_world_size=lambda: world_size,
    )


@pytest.fixture
def lance_rows(tmp_path):
    """A Lance dataset with 50 rows whose single column is the row index."""
    path = tmp_path / "rows.lance"
    lance.write_dataset(pa.table({"x": list(range(50))}), path)
    return lance.dataset(path)


def _sampled_batches(dataset, batch_size, rank=None, world_size=None):
    """Read row indices from a DistributedBatchSampler per Lance batch."""
    sampler = DistributedBatchSampler()
    if rank is None:
        batches = sampler(dataset, batch_size=batch_size)
        return [b.column("x").to_pylist() for b in batches]
    with _mock_dist(rank, world_size):
        batches = sampler(dataset, batch_size=batch_size)
        return [b.column("x").to_pylist() for b in batches]


def test_distributed_batch_sampler_shards(lance_rows):
    """Each rank reads disjoint full batches and the same number of them."""
    world_size, batch_size = 3, 4
    per_rank = [
        _sampled_batches(lance_rows, batch_size, rank, world_size)
        for rank in range(world_size)
    ]
    # 50 // (3 * 4) = 4 global batches; the last 2 rows are dropped.
    assert [len(batches) for batches in per_rank] == [4, 4, 4]
    assert all(
        len(batch) == batch_size for batches in per_rank for batch in batches
    )
    rows = [{x for batch in batches for x in batch} for batches in per_rank]
    assert sum(len(r) for r in rows) == len(set().union(*rows)) == 48
    assert per_rank[1][0] == [4, 5, 6, 7]


def test_distributed_batch_sampler_single_process(lance_rows):
    """Without a process group every row is read, as before."""
    batches = _sampled_batches(lance_rows, 16)
    assert [x for batch in batches for x in batch] == list(range(50))
    # Single world size: no rows are dropped either.
    batches = _sampled_batches(lance_rows, 16, rank=0, world_size=1)
    assert [x for batch in batches for x in batch] == list(range(50))


def test_distributed_batch_sampler_too_few_rows(lance_rows):
    """Fewer rows than one global batch cannot be sharded evenly."""
    with pytest.raises(ValueError, match="too few"):
        _sampled_batches(lance_rows, 16, rank=0, world_size=4)


@pytest.mark.parametrize("shuffle", [False, True])
def test_train_dataset_ddp_sharding(mgf_medium, tmp_path, shuffle):
    """Under DDP each rank trains on its own spectra, in equal numbers."""
    data_module = DeNovoDataModule(
        lance_dir=str(tmp_path),
        train_paths=[mgf_medium],
        train_batch_size=2,
        min_peaks=0,
        shuffle=shuffle,
    )
    data_module.setup(stage="fit")
    assert data_module._get_n_spectra(data_module.train_dataset) == 7

    # Iterate the dataset directly: the DataLoader would try to broadcast
    # its shuffle seed over the mocked process group.
    spectra = {}
    for rank in (0, 1):
        with _mock_dist(rank, 2):
            spectra[rank] = [
                scan_id
                for batch in data_module.train_dataset
                for scan_id in batch["scan_id"]
            ]
    # 7 // (2 ranks * 2) = 1 batch of 2 spectra per rank.
    assert len(spectra[0]) == len(spectra[1]) == 2
    assert not set(spectra[0]) & set(spectra[1])

    # Without a process group all spectra are used.
    single = [
        scan_id
        for batch in data_module.train_dataset
        for scan_id in batch["scan_id"]
    ]
    assert sorted(single) == sorted(f"index={i}" for i in range(7))


def test_spectrum_id_mzml(mzml_small, tmp_path):
    """Test that spectra from mzML files are specified by their scan id."""
    mzml_small2 = tmp_path / "mzml_small2.mzml"
    shutil.copy(mzml_small, mzml_small2)
    data_module = DeNovoDataModule(
        lance_dir=str(tmp_path),
        test_paths=[mzml_small, mzml_small2],
        min_peaks=0,
        shuffle=False,
    )
    data_module.setup(stage="test", annotated=False)

    dataset = data_module.test_dataset
    for i, (filename, scan_id) in enumerate(
        [
            (mzml_small, "scan=17"),
            (mzml_small, "merged=11 frame=12 scanStart=763 scanEnd=787"),
            (mzml_small2, "scan=17"),
            (mzml_small2, "merged=11 frame=12 scanStart=763 scanEnd=787"),
        ]
    ):
        assert dataset[i]["peak_file"][0] == filename.name
        assert dataset[i]["scan_id"][0] == scan_id


def test_spectrum_preprocessing(tmp_path, mgf_small):
    """
    Test the spectrum preprocessing function.
    """
    min_peaks, max_peaks = 0, 100
    min_mz, max_mz = 0, 2000
    min_intensity = 0
    remove_precursor_tol = 0
    max_charge = 4

    # Test number of peaks filtering.
    min_peaks, max_peaks = 20, 50
    # One spectrum removed with too few peaks.
    total_spectra = 1
    dataloader = DeNovoDataModule(
        str(tmp_path),
        test_paths=str(mgf_small),
        min_peaks=min_peaks,
        max_peaks=max_peaks,
        min_mz=min_mz,
        max_mz=max_mz,
        min_intensity=min_intensity,
        remove_precursor_tol=remove_precursor_tol,
        max_charge=max_charge,
    )
    dataloader.setup("test", annotated=False)
    assert dataloader.test_dataset.n_spectra == total_spectra
    for spec in dataloader.test_dataset:
        mz = (mz := spec["mz_array"][0])[torch.nonzero(mz)]
        assert min_peaks <= mz.shape[0] <= max_peaks
    min_peaks, max_peaks = 0, 100

    # Test m/z range filtering.
    min_mz, max_mz = 200, 600
    # All spectra retained.
    total_spectra = 2
    dataloader = DeNovoDataModule(
        str(tmp_path),
        test_paths=str(mgf_small),
        min_peaks=min_peaks,
        max_peaks=max_peaks,
        min_mz=min_mz,
        max_mz=max_mz,
        min_intensity=min_intensity,
        remove_precursor_tol=remove_precursor_tol,
        max_charge=max_charge,
    )
    dataloader.setup("test", annotated=False)
    assert dataloader.test_dataset.n_spectra == total_spectra
    for spec in dataloader.test_dataset:
        mz = (mz := spec["mz_array"][0])[torch.nonzero(mz)]
        assert mz.min() >= min_mz
        assert mz.max() <= max_mz
    min_mz, max_mz = 0, 2000

    # Test charge filtering.
    max_charge = 2
    # One spectrum removed with too high charge.
    total_spectra = 1
    dataloader = DeNovoDataModule(
        str(tmp_path),
        test_paths=str(mgf_small),
        min_peaks=min_peaks,
        max_peaks=max_peaks,
        min_mz=min_mz,
        max_mz=max_mz,
        min_intensity=min_intensity,
        remove_precursor_tol=remove_precursor_tol,
        max_charge=max_charge,
    )
    dataloader.setup("test", annotated=False)
    assert dataloader.test_dataset.n_spectra == total_spectra
    for spec in dataloader.test_dataset:
        assert spec["precursor_charge"] <= max_charge
    max_charge = 4


def test_data_module_per_file_validation(mgf_small, tmp_path):
    """DeNovoDataModule creates one dataset per validation/tracking file."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)
    data_module = DeNovoDataModule(
        lance_dir=str(tmp_path),
        valid_paths=[mgf_small, mgf_small2],
        tracking_paths=[mgf_small],
        min_peaks=0,
        shuffle=False,
    )
    data_module.setup()

    assert len(data_module.valid_datasets) == 2
    assert len(data_module.tracking_datasets) == 1
    assert data_module.n_main_loaders == 2
    assert data_module.val_stems == [
        mgf_small.stem,
        mgf_small2.stem,
        f"{mgf_small.stem}_1",
    ]
    loaders = data_module.val_dataloader()
    assert len(loaders) == 3
