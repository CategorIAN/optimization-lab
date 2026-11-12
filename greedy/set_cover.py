from typing import Any
from tail_recursive import tail_recursive


def greedy_set_cover(collection: Any) -> Any:
    """Execute ``greedy_set_cover``.

    Args:
        collection: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    universe = set().union(*collection.values())

    @tail_recursive
    def go(covered: Any, selected: Any) -> Any:
        """Execute ``go``.

        Args:
            covered: Input accepted by this function.
            selected: Input accepted by this function.

        Returns:
            Result produced by this function.
        """
        if covered == universe:
            return selected

        key = max(
            collection,
            key=lambda k: len(collection[k] - covered)
        )

        return go.tail_call(
            covered | collection[key],
            selected + [key]
        )

    return go(set(), [])