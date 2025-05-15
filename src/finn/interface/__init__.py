import os

IS_POSIX = os.name == "posix"
DEBUG = "FINN_DEBUG" in os.environ.keys() and os.environ["FINN_DEBUG"] == "1"
