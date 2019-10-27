import logging


def return_logger(name: str) -> logging.Logger:
    # Set log message format

    logger = logging.getLogger(name)

    if not len(logger.handlers):
        log_formatter = logging.Formatter('%(asctime)-s: %(levelname)-s %(message)s')
        # set logging level
        logger.setLevel(logging.DEBUG)

        # Direct logs to stdout
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    return logger
