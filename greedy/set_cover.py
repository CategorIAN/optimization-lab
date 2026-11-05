from tail_recursive import tail_recursive


def greedy_set_cover(collection):
    universe = set().union(*collection.values())

    @tail_recursive
    def go(covered, selected):
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