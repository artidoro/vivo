import logging
import os

def setup_logging(logger_name='vivo_logger', path='log'):
    """
    Sets up logging and formatter.
    This will log both to a file `log.log` in a directory specified by `path`
    and to the console. If the directory does not exist it will create the directory.
    The level of logging is set to DEBUG.
    Example usage:
        logger = logging.getLogger('vivo_logger')
        logger.info('Log message goes here')
    """
    # Initialize logging.
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Log to file.
    fh = logging.FileHandler(os.path.join(path, 'log.log'))
    fh.setLevel(logging.DEBUG)
    # Log to command line.
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def add_logger(logger_name='vivo_logger', path='log'):
    """
    Adds logger to file.
    """
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(path, 'log.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    return logger.addHandler(fh)
