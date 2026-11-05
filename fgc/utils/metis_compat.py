"""Drop-in replacement for the `metis` package API used by the old code, backed by pymetis."""
import pymetis


def part_graph(adjacency, nparts=2, seed=0, **_):
    opts = pymetis.Options(seed=seed)
    cuts, parts = pymetis.part_graph(nparts, adjacency=[list(map(int, a)) for a in adjacency], options=opts)
    return cuts, list(parts)
