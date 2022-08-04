import pytest
from casanovo.denovo import train, evaluate, denovo, Spec2Pep, DeNovoDataModule
import os
import yaml
from depthcharge.data import AnnotatedSpectrumIndex, SpectrumIndex
import pytorch_lightning as pl
from os.path import isfile


def test_denovo_casanovo(mgf_small):

    """Load up a basic config"""
    abs_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../casanovo/config.yaml",
    )
    with open(abs_path) as f:
        config = yaml.safe_load(f)

    """Ensure that the test doesn't raise any system overdraw issues in terms of resources"""
    config["gpus"] = []
    config["num_workers"] = 0
    output_path = os.path.dirname(os.path.abspath(__file__))

    """Running a small mgf through an untrained model just to test that the model output is valid in terms of format"""
    tabula_rasa_model = Spec2Pep(
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        dim_intensity=config["dim_intensity"],
        custom_encoder=config["custom_encoder"],
        max_length=config["max_length"],
        residues=config["residues"],
        max_charge=config["max_charge"],
        n_log=config["n_log"],
        tb_summarywriter=config["tb_summarywriter"],
        warmup_iters=config["warmup_iters"],
        max_iters=config["max_iters"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        out_filename=output_path,
    )
    from pathlib import Path

    mgf_small = Path(mgf_small).parents[0]
    mgf_files = [
        mgf_small / f
        for f in os.listdir(mgf_small)
        if (mgf_small / f).suffix.lower() == ".mgf"
    ]
    index = SpectrumIndex(
        os.path.join(os.getcwd(), config["test_annot_spec_idx_path"]),
        mgf_files,
        overwrite=config["test_spec_idx_overwrite"],
    )

    loaders = DeNovoDataModule(
        test_index=index,
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        min_intensity=config["min_intensity"],
        fragment_tol_mass=config["fragment_tol_mass"],
        num_workers=config["num_workers"],
        batch_size=config["test_batch_size"],
    )

    loaders.setup(stage="test", annotated=False)

    # Create Trainer object
    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        logger=config["logger"],
        gpus=config["gpus"],
        max_epochs=config["max_epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
    )

    trainer.test(tabula_rasa_model, loaders.test_dataloader())

    """If the output exists then the test has passed"""
    if isfile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "casanovo_output.csv"
        )
    ):
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "casanovo_output.csv",
            )
        )
        assert True
