"""Training and testing functionality for the de novo peptide sequencing model"""
import pytest
import logging, importlib, os
from pathlib import Path
import pytorch_lightning as pl
from depthcharge.data import AnnotatedSpectrumIndex, SpectrumIndex
from casanovo.denovo import DeNovoDataModule, Spec2Pep
import yaml

def train(train_data_path, val_data_path, model_path, config):
    """
    Train a Casanovo model with options specified in config.py.

    :param train_data_path: Path to .mgf files this model will be trained on
    :type train_data_path: str
    :param val_data_path: Path to .mgf files this model will be validated on
    :type val_data_path: str
    :param model_path: Path to .ckpt file to continue training (if needed)
    :type model_path: str
    :param config: Config dict passed in though casanovo
    :type config: dict 
    """ 
        
    #Set random seed across PyTorch, numpy and python.random
    pl.utilities.seed.seed_everything(seed=config['random_seed'], workers=True)

    #Index training and validation data
    train_data_path = Path(train_data_path)
    if train_data_path.is_file():
        raise FileNotFoundError(f'train_data_path expects directory path but file path was provided instead')    
    train_mgf_files = [train_data_path / f for f in os.listdir(train_data_path) if train_data_path/f.suffix.lower() == ".mgf"]
    val_data_path = Path(val_data_path)
    if val_data_path.is_file():
        raise FileNotFoundError(f'val_data_path expects directory path but file path was provided instead')        
    val_mgf_files = [val_data_path/f for f in os.listdir(val_data_path) if (val_data_path/f).suffix.lower() == ".mgf"]   
    
    train_index = AnnotatedSpectrumIndex(os.path.join(os.getcwd(),config['train_annot_spec_idx_path']), train_mgf_files, overwrite=config['train_spec_idx_overwrite'])
    val_index = AnnotatedSpectrumIndex(os.path.join(os.getcwd(),config['val_annot_spec_idx_path']), val_mgf_files, overwrite=config['val_spec_idx_overwrite'])

    #Initialize data loaders
    train_loader = DeNovoDataModule(
        train_index=train_index,
        n_peaks=config['n_peaks'],
        min_mz=config['min_mz'],
        max_mz=config['max_mz'],
        min_intensity=config['min_intensity'],
        fragment_tol_mass=config['fragment_tol_mass'],        
        preprocess_spec=config['preprocess_spec'],
        num_workers=config['num_workers'],
        batch_size=config['train_batch_size']
    )

    val_loader = DeNovoDataModule(
        valid_index=val_index,
        n_peaks=config['n_peaks'],
        min_mz=config['min_mz'],
        max_mz=config['max_mz'],
        min_intensity=config['min_intensity'],
        fragment_tol_mass=config['fragment_tol_mass'],       
        preprocess_spec=config['preprocess_spec'],
        num_workers=config['num_workers'],
        batch_size=config['val_batch_size']
    )

    train_loader.setup()
    val_loader.setup()

    # Initialize the model
    if config['train_from_scratch'] == True:
        model = Spec2Pep(
            dim_model=config['dim_model'],
            n_head=config['n_head'],
            dim_feedforward=config['dim_feedforward'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],
            dim_intensity=config['dim_intensity'],
            custom_encoder=config['custom_encoder'],
            max_length=config['max_length'],
            residues=config['residues'],
            max_charge=config['max_charge'],        
            n_log=config['n_log'],
            tb_summarywriter=config['tb_summarywriter'],
            warmup_iters = config['warmup_iters'],
            max_iters = config['max_iters'],
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        ) 

    else:
        model_path = Path(model_path)
        if model_path.is_dir():
            raise FileNotFoundError(f'model_path expects file path but directory path was provided instead')        
        model = Spec2Pep().load_from_checkpoint(
        model_path,
        dim_model=config['dim_model'],
        n_head=config['n_head'],
        dim_feedforward=config['dim_feedforward'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        dim_intensity=config['dim_intensity'],
        custom_encoder=config['custom_encoder'],
        max_length=config['max_length'],
        residues=config['residues'],
        max_charge=config['max_charge'],        
        n_log=config['n_log'],
        tb_summarywriter=config['tb_summarywriter'],
        warmup_iters = config['warmup_iters'],
        max_iters = config['max_iters'],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )



    #Create Trainer object and checkpoint callback to save model
    if config['save_model'] == True:

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=config['model_save_folder_path'],
            save_weights_only=config['save_weights_only'],
            filename='{epoch}',
            every_n_epochs=config['every_n_epochs'],
            save_top_k=-1
        )

        trainer = pl.Trainer(
            accelerator=config['accelerator'],
            logger=config['logger'],
            gpus=config['gpus'],
            max_epochs=config['max_epochs'],
            num_sanity_val_steps=config['num_sanity_val_steps'],
            callbacks=[checkpoint_callback]
        )    

    else:

        trainer = pl.Trainer(
            accelerator=config['accelerator'],
            logger=config['logger'],
            gpus=config['gpus'],
            max_epochs=config['max_epochs'],
            num_sanity_val_steps=config['num_sanity_val_steps']
        )
        
    #Train the model
    trainer.fit(model, train_loader.train_dataloader(), val_loader.val_dataloader())

def evaluate(test_data_path, model_path, config):
    """
    Run inference a pre-trained Casanovo model with evaluation and using options specified in config.py.

    :param test_data_path: Path to .mgf files this model will be tested on
    :type test_data_path: str
    :param model_path: Path to .ckpt file to evaluate
    :type model_path: str
    :param config: Config dict passed in though casanovo
    :type config: dict 
    """
        
    # Initialize the pre-trained model
    model_path = Path(model_path)
    if model_path.is_dir():
        raise FileNotFoundError(f'model_path expects file path but directory path was provided instead')
    model_trained = Spec2Pep().load_from_checkpoint(
    model_path,
    dim_model=config['dim_model'],
    n_head=config['n_head'],
    dim_feedforward=config['dim_feedforward'],
    n_layers=config['n_layers'],
    dropout=config['dropout'],
    dim_intensity=config['dim_intensity'],
    custom_encoder=config['custom_encoder'],
    max_length=config['max_length'],
    residues=config['residues'],
    max_charge=config['max_charge'],       
    n_log=config['n_log'],      
)
    #Index test data
    test_data_path = Path(test_data_path)
    if test_data_path.is_file():
        raise FileNotFoundError(f'test_data_path expects directory path but file path was provided instead')        
    mgf_files = [test_data_path/f for f in os.listdir(test_data_path) if (test_data_path/f).suffix.lower() == ".mgf"]
    index = AnnotatedSpectrumIndex(os.path.join(os.getcwd(),config['test_annot_spec_idx_path']), mgf_files, overwrite=config['test_spec_idx_overwrite'])
    
    #Initialize the data loader
    loaders=DeNovoDataModule(
        test_index=index,
        n_peaks=config['n_peaks'],
        min_mz=config['min_mz'],
        max_mz=config['max_mz'],
        min_intensity=config['min_intensity'],
        fragment_tol_mass=config['fragment_tol_mass'],
        preprocess_spec=config['preprocess_spec'],
        num_workers=config['num_workers'],
        batch_size=config['test_batch_size']
    )
    
    loaders.setup(stage='test', annotated=True)

    #Create Trainer object
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        logger=config['logger'],
        gpus=config['gpus'],
        max_epochs=config['max_epochs'],
        num_sanity_val_steps=config['num_sanity_val_steps']
    )
    
    #Run test
    trainer.validate(model_trained, loaders.test_dataloader())
    
def denovo(test_data_path, model_path, config, output_path):
    """
    Run inference with a pre-trained Casanovo model without evaluation and using options specified in config.py.

    :param test_data_path: Path to .mgf files this model will be tested on
    :type test_data_path: str
    :param model_path: Path to .ckpt file to evaluate
    :type model_path: str
    :param config: Config dict passed in though casanovo
    :type config: dict 
    :param output_path: Path to output de novo sequencing
    :type output_path: str
    """

    # Initialize the pre-trained model
    model_path = Path(model_path)
    if model_path.is_dir():
        raise FileNotFoundError(f'model_path expects file path but directory path was provided instead')
    model_trained = Spec2Pep().load_from_checkpoint(
    model_path,
    dim_model=config['dim_model'],
    n_head=config['n_head'],
    dim_feedforward=config['dim_feedforward'],
    n_layers=config['n_layers'],
    dropout=config['dropout'],
    dim_intensity=config['dim_intensity'],
    custom_encoder=config['custom_encoder'],
    max_length=config['max_length'],
    residues=config['residues'],
    max_charge=config['max_charge'],       
    n_log=config['n_log'],
    output_path=output_path        
)
    #Index test data
    test_data_path = Path(test_data_path)
    if test_data_path.is_file():
        raise FileNotFoundError(f'test_data_path expects directory path but file path was provided instead')    
    mgf_files = [test_data_path/f for f in os.listdir(test_data_path) if (test_data_path/f).suffix.lower() == ".mgf"]
    index = SpectrumIndex(os.path.join(os.getcwd(),config['test_annot_spec_idx_path']), mgf_files, overwrite=config['test_spec_idx_overwrite'])

    #Initialize the data loader
    loaders=DeNovoDataModule(
        test_index=index,
        n_peaks=config['n_peaks'],
        min_mz=config['min_mz'],
        max_mz=config['max_mz'],
        min_intensity=config['min_intensity'],
        fragment_tol_mass=config['fragment_tol_mass'],
        preprocess_spec=config['preprocess_spec'],
        num_workers=config['num_workers'],
        batch_size=config['test_batch_size']
    )
    
    loaders.setup(stage='test', annotated=False)

    #Create Trainer object
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        logger=config['logger'],
        gpus=config['gpus'],
        max_epochs=config['max_epochs'],
        num_sanity_val_steps=config['num_sanity_val_steps']
    )
    
    #Run model without evaluation
    trainer.test(model_trained, loaders.test_dataloader())
