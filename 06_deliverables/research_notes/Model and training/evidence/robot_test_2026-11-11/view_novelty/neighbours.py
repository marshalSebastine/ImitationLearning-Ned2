"""
Nearest-neighbour search between picture descriptors, with the option of leaving out the reference pictures that
share a group with the query picture (its own episode, or its own sheet): the yardstick of the view-novelty check.

The distance is the cosine distance, 1 minus the cosine of the angle between two descriptors: 0 for the same
direction, larger for less alike. The rows handed in must already have length 1 (``common.unit_rows``).
"""

from typing import Dict, Optional, Tuple

import numpy
import torch

NEIGHBOUR_PARAMETERS = {
    "count": 5,
    "chunk_rows": 512,
    "threads": 2
}



def nearest_rows(
        query: numpy.ndarray,
        reference: numpy.ndarray,
        exclusions: Optional[Dict[str, Optional[Tuple[numpy.ndarray, numpy.ndarray]]]] = None,
        parameters: Optional[Dict] = None
) -> Dict[str, Dict[str, numpy.ndarray]]:
    """
    Find the nearest reference rows of every query row, once per exclusion rule.

    Parameters
    ----------
    query : numpy.ndarray
        (queries, size) unit rows.
    reference : numpy.ndarray
        (references, size) unit rows.
    exclusions : dict, optional
        Rule name -> ``(query_groups, reference_groups)``: a reference row is left out for a query row when their
        group numbers are equal (for example the episode number, so a picture is never compared with its own
        episode). ``None`` as the value leaves nothing out. By default one rule "all" that leaves nothing out.
    parameters : dict, optional
        Overrides of ``NEIGHBOUR_PARAMETERS``: "count" (how many neighbours), "chunk_rows" and "threads".

    Returns
    -------
    dict of str to dict
        Per rule: "distance" (queries, count), the cosine distances in rising order, and "index" (queries, count),
        the reference rows they belong to.
    """

    settings = dict(NEIGHBOUR_PARAMETERS)
    settings.update(parameters or {})
    exclusions = exclusions or {"all": None}
    torch.set_num_threads(settings["threads"])
    count = settings["count"]
    reference_tensor = torch.from_numpy(numpy.ascontiguousarray(reference, dtype = numpy.float32))
    results = {name: {"distance": [], "index": []} for name in exclusions}

    for start in range(0, len(query), settings["chunk_rows"]):
        stop = min(start + settings["chunk_rows"], len(query))
        similarity = torch.from_numpy(numpy.ascontiguousarray(query[start:stop], dtype = numpy.float32)) @ reference_tensor.T
        for name, groups in exclusions.items():
            masked = similarity
            if groups is not None:
                query_groups, reference_groups = groups
                is_left_out = torch.from_numpy(numpy.asarray(query_groups[start:stop])[:, None] == numpy.asarray(reference_groups)[None, :])
                masked = similarity.masked_fill(is_left_out, -2.0)
            best = torch.topk(masked, count, dim = 1)
            results[name]["distance"].append((1.0 - best.values).numpy())
            results[name]["index"].append(best.indices.numpy())

    return {name: {key: numpy.concatenate(parts) for key, parts in result.items()} for name, result in results.items()}



def percentile_rank(
        values: numpy.ndarray,
        yardstick: numpy.ndarray
) -> numpy.ndarray:
    """
    Return, for every value, the share (0 to 100) of the yardstick values that are smaller.

    Parameters
    ----------
    values : numpy.ndarray
        The numbers to place.
    yardstick : numpy.ndarray
        The distribution they are placed in (for example the demonstrations' own distances).

    Returns
    -------
    numpy.ndarray
        Percentile ranks; 50 means "like a typical yardstick value", 100 "larger than all of them".
    """

    ordered = numpy.sort(numpy.asarray(yardstick, dtype = float))
    return 100.0 * numpy.searchsorted(ordered, numpy.asarray(values, dtype = float), side = "left") / max(len(ordered), 1)
