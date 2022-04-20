"""The command line entry point for casanovo"""
import click, logging
import yaml
import os
from casanovo.denovo import train, test_evaluate, test_denovo

#Required options
@click.command()
@click.option("--mode", required=True, default='eval', help="Choose on a high level what the program will do. \"train\" will train a model from scratch or continue training a pre-trained model. \"eval\" will evaluate de novo sequencing performance of a pre-trained model (peptide annotations are needed for spectra). \"denovo\" will run de novo sequencing without evaluation (specificy directory path for output csv file with de novo sequences).", type=click.Choice(['train', 'eval', 'denovo']))
@click.option("--model_path", required=True, help="Specify path to pre-trained model weights (.ckpt file) for testing or to continue to train.", type=click.Path(exists=True, dir_okay=False, file_okay=True))
#Base options
@click.option("--train_data_path", help="Specify path to .mgf files to be used as training data", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--val_data_path", help="Specify path to .mgf files to be used as validation data", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--test_data_path", help="Specify path to .mgf files to be used as test data", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--config_path", help="Specify path to custom config file which includes data and model related options. If not included, the default config.yaml will be used.", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.option("--output_path", help="Specify path to output de novo sequences. Output format is .csv", type=click.Path(exists=True, dir_okay=True, file_okay=False))
#De Novo sequencing options
@click.option("--preprocess_spec", help="True if spectra data should be preprocessed, False if using preprocessed data.", type=click.BOOL)
@click.option("--num_workers", help="Number of workers to use for spectra reading.", type=click.INT)
@click.option("--gpus", help="Specify gpus for usage. For multiple gpus, use format: --gpus=0 --gpus=1 --gpus=2... etc. etc.", type=click.INT, multiple=True)

def main(
    #Req + base vars
    mode,
    model_path,
    train_data_path,
    val_data_path,
    test_data_path,
    config_path,
    output_path,
    #De Novo vars
    preprocess_spec,
    num_workers,
    gpus
):
    """
    The command line function for casanovo. De Novo Mass Spectrometry Peptide Sequencing with a Transformer Model.
    
    \b
    Training option requirements:
    mode, model_path, train_data_path, val_data_path, config_path
    
    \b
    Evaluation option requirements:
    mode, model_path, test_data_path, config_path
    
    \b
    De Novo option requirements:
    mode, model_path, test_data_path, config_path, output_path
    """
    logging.basicConfig(
    level=logging.INFO, 
    format="%(levelname)s: %(message)s",
)
    if config_path == None:
        abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        with open(abs_path) as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    config['preprocess_spec'] = preprocess_spec
    config['num_workers'] = num_workers
    config['gpus'] = gpus
    if mode == 'train':
        
        logging.info('Training Casanovo...')
        train(train_data_path, val_data_path, model_path, config)
        
    elif mode == 'eval':
        
        logging.info('Evaluating Casanovo...')
        test_evaluate(test_data_path, model_path, config)

    elif mode == 'denovo':
        
        logging.info('De novo sequencing with Casanovo...')
        test_denovo(test_data_path, model_path, config, output_path)  
        
    pass


if __name__ == "__main__":
    main()

