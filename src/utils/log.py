import logger


class Logger(logger):
    """
    Custom logger class that extends the base logger.
    This class can be used to log messages with different levels of severity.
    """

    def __init__(self, name: str = "default_logger"):
        super().__init__(name)

    def log_info(self, message: str):
        self.info(message)

    def log_warning(self, message: str):
        self.warning(message)

    def log_error(self, message: str):
        self.error(message)