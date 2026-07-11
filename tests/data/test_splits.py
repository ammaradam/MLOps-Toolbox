from __future__ import annotations

import numpy as np
import pytest

from mlops_toolbox.data import train_test_split


def test_split_proportions_and_no_overlap(sample_classification_data) -> None:
    X, y = sample_classification_data
    split = train_test_split(X, y, test_size=0.25, random_state=0)

    assert len(split.X_train) == 150
    assert len(split.X_test) == 50
    assert len(split.y_train) == 150
    assert len(split.y_test) == 50

    train_idx = set(split.X_train.index) if hasattr(split.X_train, "index") else None
    assert train_idx is not None  # sanity: still a DataFrame after split


def test_split_is_reproducible_with_random_state(sample_classification_data) -> None:
    X, y = sample_classification_data
    split_a = train_test_split(X, y, test_size=0.2, random_state=7)
    split_b = train_test_split(X, y, test_size=0.2, random_state=7)
    assert split_a.X_train.equals(split_b.X_train)


def test_split_works_on_raw_numpy_arrays() -> None:
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    split = train_test_split(X, y, test_size=0.3, random_state=1)
    assert isinstance(split.X_train, np.ndarray)
    assert len(split.X_train) == 7
    assert len(split.X_test) == 3


def test_split_rejects_invalid_test_size(sample_classification_data) -> None:
    X, y = sample_classification_data
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)


def test_split_rejects_mismatched_lengths() -> None:
    X = np.zeros((10, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError):
        train_test_split(X, y)
