"""Test for api.distribute_samples."""

import polars as pl

from perseus.api.distribute_samples import distribute_samples


def test_distribute_samples_yields_partitions_with_index() -> None:
    samples = pl.DataFrame({"client_id": ["c1", "c2", "c3", "c1"], "value": [1, 2, 3, 4]})
    out = list(distribute_samples(samples))

    assert len(out) >= 1
    seen_clients = set()
    for pid, part in out:
        assert isinstance(pid, int)
        # partition_id is removed, _index is added
        assert "partition_id" not in part.schema
        assert "_index" in part.schema
        assert part["_index"].to_list() == list(range(len(part)))
        seen_clients.update(part["client_id"].to_list())
    # all clients are distributed
    assert seen_clients == {"c1", "c2", "c3"}
