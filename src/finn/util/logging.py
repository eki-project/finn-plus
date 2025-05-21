import logging
from rich.logging import RichHandler

log = logging.getLogger("finn_logger")
log.addHandler(RichHandler(show_path=False, show_time=False))
