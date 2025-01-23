from typing import Any

__author__ = "keviny2"
__copyright__ = "keviny2"
__licence__ = "MIT"


def is_list_of_type(x: Any, target_type: callable) -> bool:
    """Checks if `x` is a list of `target_type`.

    Args:
        x (Any): Any object.
        target_type (callable): Type to check for, e.g. str, int

    Returns:
        bool: True if `x` is list and all values are of the same type.
    """
    return isinstance(x, (list, tuple)) and all(
        isinstance(item, target_type) for item in x
    )
