import sys
import logging
import logging.handlers as handlers


LOAN_LOGGER = logging.getLogger('loan_logger')
LOAN_LOGGER.setLevel(logging.INFO)

loan_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
loan_logHandler = handlers.TimedRotatingFileHandler('loan_logs.log', when='H', interval=1, backupCount=2)
loan_logHandler.setLevel(logging.INFO)
loan_logHandler.setFormatter(loan_formatter)
LOAN_LOGGER.addHandler(loan_logHandler)


def log_and_stop(logger, message):
    """
    Save the exception message into the logger and stop the script.
    """
    logger.exception(message)
    sys.exit(message)
