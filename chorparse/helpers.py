from typing import Any, Generator, Iterable, List, Tuple


def select(
    iterables: List[Tuple[Any, Iterable[Any]]], i=0, result: List = []
) -> Generator[List[Tuple[Any, Any]], None, None]:
    """
    Generates all ways of selecting single elements 
    out of a sequence of iterables.

    >>> select([('A',[1]),('B',[1,2])])
    [[('A',1),('B',1)], [('A',1),('B',2)]]
    """
    if i >= len(iterables):
        yield result.copy()
    else:
        p, ls = iterables[i]
        for x in ls:
            result.append((p, x))
            yield from select(iterables=iterables, i=i + 1, result=result)
            result.pop()
