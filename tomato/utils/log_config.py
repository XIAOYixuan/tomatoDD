import logging

# Configure the logging module
logging.basicConfig(format='%(filename)s:%(levelname)s:%(lineno)d:%(message)s', level=logging.DEBUG)

# Get the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_logger_output(save_dir: str):
    # redirect the logger output to the save_dir's train.log
    import os
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(filename)s:%(levelname)s:%(lineno)d:%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)