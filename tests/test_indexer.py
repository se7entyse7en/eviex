import itertools
from datetime import datetime
from datetime import timezone
from typing import List
from typing import Tuple

import numpy as np
import pytest

from eviex.indexer import LayerLevel
from eviex.indexer import MemoryIndexer


mock_data_none_granularity = [
    {
        "timestamp": datetime(1970, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["a"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 0, 0, 0, 1).replace(tzinfo=timezone.utc),
        "values": ["b"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 1, 0, 0, 2).replace(tzinfo=timezone.utc),
        "values": ["c"],
    },
    {
        "timestamp": datetime(1970, 1, 2, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["d"],
    },
    {
        "timestamp": datetime(1970, 1, 2, 0, 0, 0, 1).replace(tzinfo=timezone.utc),
        "values": ["e"],
    },
    {
        "timestamp": datetime(1970, 1, 2, 0, 0, 0, 2).replace(tzinfo=timezone.utc),
        "values": ["f"],
    },
    {
        "timestamp": datetime(1971, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["g"],
    },
    {
        "timestamp": datetime(1971, 1, 1, 0, 0, 0, 1).replace(tzinfo=timezone.utc),
        "values": ["h"],
    },
    {
        "timestamp": datetime(1971, 1, 1, 0, 0, 0, 2).replace(tzinfo=timezone.utc),
        "values": ["i"],
    },
]


mock_data_small_granularity = [
    {
        "timestamp": datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["a"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 0, 45, 0).replace(tzinfo=timezone.utc),
        "values": ["b"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 1, 15, 0).replace(tzinfo=timezone.utc),
        "values": ["c"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 3, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["d"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 3, 15, 0).replace(tzinfo=timezone.utc),
        "values": ["e"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 3, 30, 0).replace(tzinfo=timezone.utc),
        "values": ["f"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 3, 45, 0).replace(tzinfo=timezone.utc),
        "values": ["g"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 4, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["h"],
    },
    {
        "timestamp": datetime(1970, 1, 1, 4, 45, 0).replace(tzinfo=timezone.utc),
        "values": ["i"],
    },
]

mock_data_big_granularity = [
    {
        "timestamp": datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["a"],
    },
    {
        "timestamp": datetime(1970, 2, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["b"],
    },
    {
        "timestamp": datetime(1970, 3, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["c"],
    },
    {
        "timestamp": datetime(1970, 4, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["d"],
    },
    {
        "timestamp": datetime(1970, 7, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["e"],
    },
    {
        "timestamp": datetime(1970, 9, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["f"],
    },
    {
        "timestamp": datetime(1971, 3, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["g"],
    },
    {
        "timestamp": datetime(1971, 11, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["h"],
    },
    {
        "timestamp": datetime(1971, 12, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        "values": ["i"],
    },
]


def generate_levels_combinations(
    *min_levels: LayerLevel,
) -> List[Tuple[LayerLevel, LayerLevel]]:
    def gen(ml):
        return [
            (ml, LayerLevel.get(ll_value))
            for ll_value in LayerLevel.levels().keys()
            if ml.value < ll_value
        ]

    return list(itertools.chain.from_iterable(gen(ml) for ml in min_levels))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "min_level, max_level",
    generate_levels_combinations(LayerLevel.NONE, LayerLevel.SECOND, LayerLevel.MINUTE),
)
@pytest.mark.parametrize(
    "date_from, date_to, expected",
    [
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 5, 0, 0).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1969, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1969, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1971, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1972, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1970, 1, 1, 0, 10, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 3, 40, 0).replace(tzinfo=timezone.utc),
            np.array(["b", "c", "d", "e", "f"]),
        ),
        (
            datetime(1970, 1, 1, 0, 40, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 50, 0).replace(tzinfo=timezone.utc),
            np.array(["b"]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 4, 45, 0).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g", "h"]),
        ),
    ],
)
async def test_get_with_no_precision_loss_small_granularity(
    min_level,
    max_level,
    date_from,
    date_to,
    expected,
):
    """Test query with interval granularity not smaller than the mininum layer level.

    In this case we don't have any precision loss.

    """
    indexer = MemoryIndexer(min_level=min_level, max_level=max_level)
    await indexer.load(mock_data_small_granularity)
    actual = indexer.get(date_from, date_to)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "min_level, max_level",
    generate_levels_combinations(LayerLevel.HOUR, LayerLevel.DAY, LayerLevel.MONTH),
)
@pytest.mark.parametrize(
    "date_from, date_to, expected",
    [
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1972, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1969, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1969, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1972, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1973, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1970, 2, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 10, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array(["b", "c", "d", "e", "f"]),
        ),
        (
            datetime(1970, 2, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 3, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array(["b"]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1971, 12, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g", "h"]),
        ),
    ],
)
async def test_get_with_no_precision_loss_big_granularity(
    min_level,
    max_level,
    date_from,
    date_to,
    expected,
):
    """Test query with interval granularity not smaller than the mininum layer level.

    In this case we don't have any precision loss.

    """
    indexer = MemoryIndexer(min_level=min_level, max_level=max_level)
    await indexer.load(mock_data_big_granularity)
    actual = indexer.get(date_from, date_to)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "min_level, max_level",
    generate_levels_combinations(LayerLevel.HOUR),
)
@pytest.mark.parametrize(
    "date_from, date_to, expected",
    [
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 5, 0, 0).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1969, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1969, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1971, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1972, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1970, 1, 1, 0, 10, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 3, 40, 0).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c"]),
        ),
        (
            datetime(1970, 1, 1, 0, 40, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 50, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 4, 45, 0).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g"]),
        ),
    ],
)
async def test_get_with_query_precision_loss_small_granularity(
    min_level,
    max_level,
    date_from,
    date_to,
    expected,
):
    """Test query with interval granularity smaller than the minimum layer level.

    In this case we have a precision loss due to the `date_from` and `date_to` being
    transformed to a less granular value corresponding to the minimum layer level.

    """
    indexer = MemoryIndexer(min_level=min_level, max_level=max_level)
    await indexer.load(mock_data_small_granularity)
    actual = indexer.get(date_from, date_to)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "min_level, max_level",
    generate_levels_combinations(LayerLevel.DAY),
)
@pytest.mark.parametrize(
    "date_from, date_to, expected",
    [
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1972, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1969, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1969, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1972, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1973, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1970, 2, 1, 1, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 7, 1, 3, 0, 0).replace(tzinfo=timezone.utc),
            np.array(["b", "c", "d"]),
        ),
        (
            datetime(1970, 8, 1, 0, 30, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 9, 1, 0, 50, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1970, 1, 1, 12, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1971, 11, 1, 6, 30, 0).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g"]),
        ),
    ],
)
async def test_get_with_query_precision_loss_big_granularity(
    min_level,
    max_level,
    date_from,
    date_to,
    expected,
):
    """Test query with interval granularity smaller than the minimum layer level.

    In this case we have a precision loss due to the `date_from` and `date_to` being
    transformed to a less granular value corresponding to the minimum layer level.

    """
    indexer = MemoryIndexer(min_level=min_level, max_level=max_level)
    await indexer.load(mock_data_big_granularity)
    actual = indexer.get(date_from, date_to)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "date_from, date_to, expected",
    [
        (
            datetime(1970, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1971, 1, 1, 0, 0, 0, 3).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1969, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1969, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1972, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1973, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
            np.array([]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0, 1).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 2, 0, 0, 0, 3).replace(tzinfo=timezone.utc),
            np.array(["b", "c", "d", "e", "f"]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0, 1).replace(tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 0, 0, 2).replace(tzinfo=timezone.utc),
            np.array(["b"]),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 0, 0).replace(tzinfo=timezone.utc),
            datetime(1971, 1, 1, 0, 0, 0, 2).replace(tzinfo=timezone.utc),
            np.array(["a", "b", "c", "d", "e", "f", "g", "h"]),
        ),
    ],
)
async def test_get_with_none_granularity(
    date_from,
    date_to,
    expected,
):
    """Test query with data with LayerLevel.NONE granularity."""
    indexer = MemoryIndexer(min_level=LayerLevel.NONE, max_level=LayerLevel.NONE)
    await indexer.load(mock_data_none_granularity)
    actual = indexer.get(date_from, date_to)
    np.testing.assert_array_equal(actual, expected)
