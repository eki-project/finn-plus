import os

def find_deps() -> str | None:
    """Find the deps folder containing the board files, finn-experimental, cnpy etc.
    This does NOT check for the contents of the dep folder
    Returns the path including the deps/ if found, None otherwise
    
    First searches in ~/.finn, then in $FINN_ROOT"""
    HOMEPATH = os.path.abspath(os.path.join(os.environ["HOME"], ".finn", "deps"))
    LOCALPATH = os.path.abspath(os.path.join(os.environ["FINN_ROOT"], "deps"))
    if os.path.isdir(HOMEPATH):
        return HOMEPATH
    if os.path.isdir(LOCALPATH):
        return LOCALPATH
    return None


def find_rtllib() -> str | None:
    """Find the rtllib folder 
    First searches in ~/.finn, then in $FINN_ROOT"""
    HOMEPATH = os.path.abspath(os.path.join(os.environ["HOME"], ".finn", "finn-rtllib"))
    LOCALPATH = os.path.abspath(os.path.join(os.environ["FINN_ROOT"], "deps"))
    if os.path.isdir(HOMEPATH):
        return HOMEPATH
    if os.path.isdir(LOCALPATH):
        return LOCALPATH
    return None