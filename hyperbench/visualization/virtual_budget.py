import re

def get_multiplier(optimizer: str):
    """
    If the optimizer has e.g. _x2 or _x3 in its name, this means that the optimizer has been given a multiple of the
    budget of the others in the experiment. This function will retrieve that multiplier from its name.

    Parameters
    ----------
    optimizer: str
        The name of the optimizer

    Returns
    -------
    multiplier: int
        The multiplier retrieved from the name
    """
    has_multiplied_budget = bool(re.match(".*_x\d*", optimizer))
    multiplier = 1 if not has_multiplied_budget else int(re.search("_x\d*", optimizer)[0].replace("_x", ""))
    return multiplier