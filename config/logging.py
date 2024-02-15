import logging.config

level = "DEBUG"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default_formatter": {"format": "[%(levelname)s:%(asctime)s] %(message)s"},
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "default_formatter",
        },
    },
    "loggers": {
        "basic_logger": {
            "handlers": ["stream_handler"],
            "level": level,
            "propagate": True,
        }
    },
}


def set_logger():
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("basic_logger")
    return logger
