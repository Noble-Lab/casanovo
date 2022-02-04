"""Training and testing functionality for the de novo peptide sequencing model"""
import logging
from pathlib import Path
from depthcharge.data import AnnotatedSpectrumIndex 
from casanovo.denovo import DeNovoDataModule, Spec2Pep
from casanovo import config
import pytorch_lightning as pl

def train():
    """Train a Casanovo model with options specified in config.py."""    
    
    #Set random seed across PyTorch, numpy and python.random
    pl.utilities.seed.seed_everything(seed=config.random_seed, workers=True)

    #Index training and validation data
    train_mgf_file = Path(config.train_data_path)
    val_mgf_file = Path(config.val_data_path)

    train_index = AnnotatedSpectrumIndex(config.train_annot_spec_idx_path, train_mgf_file, overwrite=config.train_spec_idx_overwrite)
    val_index = AnnotatedSpectrumIndex(config.val_annot_spec_idx_path, val_mgf_file, overwrite=config.val_spec_idx_overwrite)

    #Initialize data loaders
    train_loader = DeNovoDataModule(
        train_index=train_index,
        n_peaks=config.n_peaks,
        min_mz=config.min_mz,
        preprocess_spec=config.preprocess_spec,
        num_workers=config.num_workers,
        batch_size=config.train_batch_size
    )

    val_loader = DeNovoDataModule(
        valid_index=val_index,
        n_peaks=config.n_peaks,
        min_mz=config.min_mz,
        preprocess_spec=config.preprocess_spec,
        num_workers=config.num_workers,
        batch_size=config.val_batch_size
    )

    train_loader.setup()
    val_loader.setup()

    # Initialize the model
    if config.train_from_scratch == True:
        model = Spec2Pep(
            dim_model=config.dim_model,
            n_head=config.n_head,
            dim_feedforward=config.dim_feedforward,
            n_layers=config.n_layers,
            dropout=config.dropout,
            dim_intensity=config.dim_intensity,
            custom_encoder=config.custom_encoder,
            max_length=config.max_length,
            residues=config.residues,
            max_charge=config.max_charge,        
            n_log=config.n_log,
            tb_summarywriter=config.tb_summarywriter,
            warmup_iters = config.warmup_iters,
            max_iters = config.max_iters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        ) 

    else:
        model = Spec2Pep().load_from_checkpoint(
        config.model_full_path,
        dim_model=config.dim_model,
        n_head=config.n_head,
        dim_feedforward=config.dim_feedforward,
        n_layers=config.n_layers,
        dropout=config.dropout,
        dim_intensity=config.dim_intensity,
        custom_encoder=config.custom_encoder,
        max_length=config.max_length,
        residues=config.residues,
        max_charge=config.max_charge,        
        n_log=config.n_log,
        tb_summarywriter=config.tb_summarywriter,
        warmup_iters = config.warmup_iters,
        max_iters = config.max_iters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )



    #Create Trainer object and checkpoint callback to save model
    if config.save_model == True:

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=config.model_save_folder_path,
            save_weights_only=config.save_weights_only,
            filename='{epoch}',
            every_n_epochs=config.every_n_epochs,
            save_top_k=-1
        )

        trainer = pl.Trainer(
            strategy=config.accelerator,
            logger=config.logger,
            gpus=config.gpus,
            max_epochs=config.max_epochs,
            num_sanity_val_steps=config.num_sanity_val_steps,
            callbacks=[checkpoint_callback]
        )    

    else:

        trainer = pl.Trainer(
            strategy=config.accelerator,
            logger=config.logger,
            gpus=config.gpus,
            max_epochs=config.max_epochs,
            num_sanity_val_steps=config.num_sanity_val_steps
        )
        
    #Train the model
    trainer.fit(model, train_loader.train_dataloader(), val_loader.val_dataloader())

def test_denovo():
    """Test a pre-trained Casanovo model with options specified in config.py."""
    
    # Initialize the pre-trained model
    model_trained = Spec2Pep().load_from_checkpoint(
    config.model_full_path,
    dim_model=config.dim_model,
    n_head=config.n_head,
    dim_feedforward=config.dim_feedforward,
    n_layers=config.n_layers,
    dropout=config.dropout,
    dim_intensity=config.dim_intensity,
    custom_encoder=config.custom_encoder,
    max_length=config.max_length,
    residues=config.residues,
    max_charge=config.max_charge,       
    n_log=config.n_log,       
)
    #Index test data
    mgf_file = Path(config.test_data_path)
    index = AnnotatedSpectrumIndex(config.test_annot_spec_idx_path, mgf_file, overwrite=config.test_spec_idx_overwrite)
    
    #Initialize the data loader
    loaders=DeNovoDataModule(
        test_index=index,
        n_peaks=config.n_peaks,
        min_mz=config.min_mz,
        preprocess_spec=config.preprocess_spec,
        num_workers=config.num_workers,
        batch_size=config.test_batch_size
    )
    
    loaders.setup()

    #Create Trainer object
    trainer = pl.Trainer(
        strategy=config.accelerator,
        logger=config.logger,
        gpus=config.gpus,
        max_epochs=config.max_epochs,
        num_sanity_val_steps=config.num_sanity_val_steps
    )
    
    #Run test
    trainer.validate(model_trained, loaders.test_dataloader())


