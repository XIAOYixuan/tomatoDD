import logging

# Configure the logging module
logging.basicConfig(format='%(filename)s:%(levelname)s:%(lineno)d:%(message)s', level=logging.DEBUG)

# Get the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)