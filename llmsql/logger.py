import logging
import colorlog


class Logger:
    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
        "crit": logging.CRITICAL,
    }

    def __init__(self, level="info"):
        color_fmt = "%(log_color)s%(asctime)s %(levelname)s %(pathname)s[%(lineno)d]: %(message)s"
        color_formatter = colorlog.ColoredFormatter(
            color_fmt,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        # Create console handler
        sh = logging.StreamHandler()
        sh.setFormatter(color_formatter)

        # Create logger
        self.logger = logging.getLogger("llmsql")
        self.logger.setLevel(self.level_relations.get(level, logging.INFO))
        self.logger.addHandler(sh)

        # Prevent log propagation to avoid duplicate logs
        self.logger.propagate = False


logger = Logger(level="info").logger
