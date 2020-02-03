from datasets import EqualityDataset, PremackDataset
import numpy as np
import pytest


@pytest.fixture
def tiny_equality_dataset():
    return EqualityDataset(embed_dim=10, n_pos=2, n_neg=10, flatten=True)


def test_equality_create_pos(tiny_equality_dataset):
    result = tiny_equality_dataset._create_pos()
    assert len(result) == tiny_equality_dataset.n_pos
    for (v1, v2), label in result:
        assert label == tiny_equality_dataset.POS_LABEL
        assert np.array_equal(v1, v2)


def test_equality_create_neg(tiny_equality_dataset):
    result = tiny_equality_dataset._create_neg()
    assert len(result) == tiny_equality_dataset.n_neg
    for (v1, v2), label in result:
        assert label == tiny_equality_dataset.NEG_LABEL
        assert not np.array_equal(v1, v2)


@pytest.mark.parametrize("flatten, expected", [
    [True, (4, 20)],
    [False, (4, 2, 10)]
])
def test_flatten(flatten, expected):
    dataset = EqualityDataset(embed_dim=10, n_pos=2, n_neg=2, flatten=flatten)
    assert dataset.flatten == flatten
    X, y = dataset.create()
    result = X.shape
    assert result == expected


@pytest.mark.parametrize("cls, expected", [
    [1, 2],
    [0, 10]
])
def test_equality_create_label_dist(cls, expected):
    dataset = EqualityDataset(embed_dim=2, n_pos=2, n_neg=10)
    X, y = dataset.create()
    result = sum([1 for label in y if label == cls])
    assert result == expected


@pytest.mark.parametrize("cls, expected", [
    [1, True],
    [0, False]
])
def test_equality_create_vector_relations(cls, expected):
    dataset = EqualityDataset(embed_dim=2, n_pos=2, n_neg=2, flatten=False)
    dataset.create()
    for (v1, v2), label in dataset.data:
        if label == cls:
            rel = np.array_equal(v1, v2)
            assert rel == expected


@pytest.mark.parametrize("flatten", [True, False])
def test_equality_disjoint(flatten):
    dataset = EqualityDataset(embed_dim=2, n_pos=2, n_neg=2, flatten=flatten)
    dataset.create()
    with pytest.raises(AssertionError):
        dataset.test_disjoint(dataset)


@pytest.fixture
def tiny_premack_dataset():
    return PremackDataset(
        embed_dim=10,
        n_pos=2,
        n_neg=10,
        flatten_root=True,
        flatten_leaves=True)


def test_premack_create_same_same(tiny_premack_dataset):
    result = tiny_premack_dataset._create_same_same()
    assert len(result) == tiny_premack_dataset.n_same_same
    for (p1, p2), label in result:
        assert label == tiny_premack_dataset.POS_LABEL
        assert np.array_equal(p1[0], p1[1])
        assert np.array_equal(p2[0], p2[1])


def test_premack_create_diff_diff(tiny_premack_dataset):
    result = tiny_premack_dataset._create_diff_diff()
    assert len(result) == tiny_premack_dataset.n_diff_diff
    for (p1, p2), label in result:
        assert label == tiny_premack_dataset.POS_LABEL
        assert not np.array_equal(p1[0], p1[1])
        assert not np.array_equal(p2[0], p2[1])


def test_premack_create_same_diff(tiny_premack_dataset):
    result = tiny_premack_dataset._create_same_diff()
    assert len(result) == tiny_premack_dataset.n_same_diff
    for (p1, p2), label in result:
        assert label == tiny_premack_dataset.NEG_LABEL
        assert np.array_equal(p1[0], p1[1])
        assert not np.array_equal(p2[0], p2[1])


def test_premack_create_diff_same(tiny_premack_dataset):
    result = tiny_premack_dataset._create_diff_same()
    assert len(result) == tiny_premack_dataset.n_diff_same
    for (p1, p2), label in result:
        assert label == tiny_premack_dataset.NEG_LABEL
        assert not np.array_equal(p1[0], p1[1])
        assert np.array_equal(p2[0], p2[1])


@pytest.mark.parametrize("flatten_root, flatten_leaves", [
    [True, True],
    [True, False],
    [False, True],
    [False, False]
])
def test_premack_disjoint(flatten_root, flatten_leaves):
    dataset = PremackDataset(
        embed_dim=2, n_pos=2, n_neg=2,
        flatten_root=flatten_root,
        flatten_leaves=flatten_leaves)
    dataset.create()
    with pytest.raises(AssertionError):
        dataset.test_disjoint(dataset)


@pytest.mark.parametrize("flatten_root, flatten_leaves, expected", [
    [True, True, (4, 40)],
    [True, False, (4, 40)],
    [False, True, (4, 2, 20)],
    [False, False, (4, 2, 2, 10)]
])
def test_premack_flattening(flatten_root, flatten_leaves, expected):
    dataset = PremackDataset(
        embed_dim=10, n_pos=2, n_neg=2,
        flatten_root=flatten_root,
        flatten_leaves=flatten_leaves)
    assert dataset.flatten_root == flatten_root
    assert dataset.flatten_leaves == flatten_leaves
    X, y = dataset.create()
    result = X.shape
    assert result == expected


@pytest.mark.parametrize("cls, expected", [
    [1, 10],
    [0, 2]
])
def test_premack_create_label_dist(cls, expected):
    dataset = PremackDataset(embed_dim=2, n_pos=10, n_neg=2)
    X, y = dataset.create()
    result = sum([1 for label in y if label == cls])
    assert result == expected

@pytest.mark.parametrize("n_pos, n_neg", [
    [2, 3],
    [3, 2]
])
def test_premack_odd_size_value_error(n_pos, n_neg):
    with pytest.raises(ValueError):
        PremackDataset(embed_dim=2, n_pos=n_pos, n_neg=n_neg)
