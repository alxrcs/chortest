from pathlib import Path
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


def combine_fsa(
    input_fsa_filename: str,
    replacement_fsa_filename: str,
    output_filename: str = None,
):
    # TODO: Switch this to use a proper parser
    import re

    output_filename = output_filename or str(
        Path(input_fsa_filename).with_suffix(".fsa")
    )

    with open(input_fsa_filename, "r") as input_fsa_file, open(
        replacement_fsa_filename, "r"
    ) as replacement_fsa_file:

        parse_fsm = lambda txt: {
            f[1]: f[0] for f in re.findall(r"(\.outputs (\w+).*?\.end)", txt, re.DOTALL)
        }

        updated_fsm = parse_fsm(input_fsa_file.read())
        updated_fsm.update(parse_fsm(replacement_fsa_file.read()))

        txt = "\n\n".join(updated_fsm.values())

    Path(output_filename).parent.mkdir(exist_ok=True)

    with open(output_filename, "w") as output_fsa_file:
        output_fsa_file.write(txt)
