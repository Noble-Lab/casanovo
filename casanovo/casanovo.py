"""The command line entry point for casanovo"""
import click, logging
from casanovo.denovo import train, test_denovo


@click.command()
@click.option("--mode", default='test', help="Choose to train a model or test denovo predictions")
@click.option("--model_path", help="Specify path to pre-trained model weights for testing or to continue to train")
@click.option("--train_data_path", help="Specify path to mgf files to be used as training data")
@click.option("--val_data_path", help="Specify path to mgf files to be used as validation data")
@click.option("--test_data_path", help="Specify path to mgf files to be used as test data")
@click.option("--config_path", help="Specify path to config file which includes data and model related options")

def main(
    mode,
    model_path,
    train_data_path,
    val_data_path,
    test_data_path,
    config_path
):
    """The command line function"""
    logging.basicConfig(
    level=logging.INFO, 
    format="%(levelname)s: %(message)s",
)
    if mode == 'train':
        
        logging.info('Training Casanovo...')
        train(train_data_path, val_data_path, model_path, config_path)
        
    elif mode == 'test':
        
        logging.info('Testing Casanovo...')
        test_denovo(test_data_path, model_path, config_path)
        
    pass


if __name__ == "__main__":
    main()

