"""Tests for lightweight data utilities (no model or dataset download).

These are deliberately fast and dependency-light so they run in CI on every
push. They cover the logic most likely to break silently: the label map and
the stratified split determinism.
"""

import pandas as pd

from src import config, data


def _toy_df(n_per_class: int = 40) -> pd.DataFrame:
    """A tiny synthetic dataset with the four expected classes."""
    rows = []
    for cls in ["benign", "phishing", "malware", "defacement"]:
        for i in range(n_per_class):
            rows.append({"url": f"http://{cls}-{i}.example.com", "type": cls})
    return pd.DataFrame(rows)


def test_build_label_map_is_sorted_and_complete():
    df = _toy_df()
    label_map = data.build_label_map(df)
    # Deterministic alphabetical ordering.
    assert list(label_map) == ["benign", "defacement", "malware", "phishing"]
    # Contiguous zero-based ids.
    assert sorted(label_map.values()) == [0, 1, 2, 3]


def test_label_map_round_trip(tmp_path):
    df = _toy_df()
    label_map = data.build_label_map(df)
    data.save_label_map(label_map, tmp_path)
    loaded = data.load_label_map(tmp_path)
    assert loaded == label_map


def test_stratified_split_is_deterministic():
    df = _toy_df()
    a = data.stratified_split(df)
    b = data.stratified_split(df)
    # Same seed -> identical splits across calls.
    for split_a, split_b in zip(a, b):
        assert list(split_a.index) == list(split_b.index)


def test_stratified_split_preserves_all_classes():
    df = _toy_df()
    train_df, val_df, test_df = data.stratified_split(df)
    classes = set(df["type"])
    # Every class must appear in every split.
    assert set(train_df["type"]) == classes
    assert set(val_df["type"]) == classes
    assert set(test_df["type"]) == classes


def test_split_fractions_are_reasonable():
    df = _toy_df(n_per_class=100)  # 400 rows total
    train_df, val_df, test_df = data.stratified_split(df)
    total = len(df)
    # ~70 / 15 / 15 within a tolerance.
    assert abs(len(train_df) / total - 0.70) < 0.03
    assert abs(len(val_df) / total - 0.15) < 0.03
    assert abs(len(test_df) / total - 0.15) < 0.03


def test_config_constants_are_consistent():
    # NUM_LABELS should match the four canonical classes.
    assert config.NUM_LABELS == 4
    assert 0.0 < config.UNCERTAIN_THRESHOLD < 1.0
    assert config.MAX_SEQ_LENGTH > 0


def test_load_and_encode_splits_labels_every_split(tmp_path):
    """Regression test: every split must carry a 'label' column.

    This guards against a real bug where a caller unpacked
    ``encode_labels``'s three return values as e.g. ``_, _, test_df = ...``,
    silently leaving the discarded splits unlabeled. That crashes with
    ``KeyError: 'label'`` only once those splits reach tokenization --
    exactly the kind of mistake ``load_and_encode_splits`` exists to make
    structurally impossible, so we test it directly here.
    """
    df = _toy_df()
    csv_path = tmp_path / "toy.csv"
    df.to_csv(csv_path, index=False)

    # Case 1: no label_map given (as in training) -- one is built fresh.
    train_df, val_df, test_df, label_map = data.load_and_encode_splits(
        csv_path=csv_path
    )
    for split in (train_df, val_df, test_df):
        assert "label" in split.columns
        assert split["label"].notna().all()
    assert label_map == data.build_label_map(df)

    # Case 2: an existing label_map is passed in (as in evaluate/calibrate).
    train_df2, val_df2, test_df2, label_map2 = data.load_and_encode_splits(
        label_map, csv_path=csv_path
    )
    for split in (train_df2, val_df2, test_df2):
        assert "label" in split.columns
    assert label_map2 == label_map
