import logging
import sys

def get_logger(name: str = "trader"):
    """
    Returns a singleton-style logger that writes to stdout.
    Other modules should import and use get_logger().
    """
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(handler)
    return log
