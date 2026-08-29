from sp500_forecastability.splits import walk_forward_splits


def test_walk_forward_splits_are_ordered_and_have_a_gap() -> None:
    splits = list(walk_forward_splits(12, train_size=5, test_size=2, step=2, gap=1))

    assert splits[0] == (tuple(range(5)), (6, 7))
    assert splits[1][0][-1] == 6
    assert max(splits[0][0]) < min(splits[0][1])
    assert max(splits[0][0]) + 1 < min(splits[0][1])

