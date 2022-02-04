"""The command line entry point for casanovo"""
import sys, logging
from casanovo.denovo import train, test_denovo


def main():
    """The command line function"""
    logging.basicConfig(
    level=logging.INFO, 
    format="%(levelname)s: %(message)s",
)
    if sys.argv[1:][0] == 'train':
        
        logging.info('Training Casanovo...')
        train()
        
    elif sys.argv[1:][0] == 'test':
        
        logging.info('Testing Casanovo...')
        test_denovo()
        
    pass


if __name__ == "__main__":
    main()
