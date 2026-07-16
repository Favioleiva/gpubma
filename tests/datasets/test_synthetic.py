import numpy as np
import pandas as pd
import pytest

from gpubma.datasets.synthetic import DEFAULT_SEED, generate_panel, true_beta


def test_deterministic_regeneration():
    df1, _ = generate_panel(100, 10, 8, seed=DEFAULT_SEED)
    df2, _ = generate_panel(100, 10, 8, seed=DEFAULT_SEED)
    pd.testing.assert_frame_equal(df1, df2)


def test_identical_row_ordering():
    df, _ = generate_panel(20, 5, 3)
    expected = df.sort_values(["individual_id", "period"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


def test_unique_individual_time_keys():
    df, _ = generate_panel(30, 7, 4)
    assert not df.duplicated(subset=["individual_id", "period"]).any()


def test_balanced_panel_structure():
    df, _ = generate_panel(30, 7, 4)
    counts = df.groupby("individual_id")["period"].count()
    assert (counts == 7).all()
    assert len(df) == 30 * 7


@pytest.mark.parametrize("p", [8, 12, 30])
def test_expected_number_of_predictors(p):
    df, meta = generate_panel(10, 4, p)
    xcols = [c for c in df.columns if c.startswith("x")]
    assert len(xcols) == p == meta["n_optional_predictors"]


@pytest.mark.parametrize("p", [8, 12, 30])
def test_expected_model_count(p):
    _, meta = generate_panel(10, 4, p)
    assert meta["expected_number_of_models"] == 2**p


def test_first_eight_predictors_nested_across_sizes():
    dfs = {p: generate_panel(100, 10, p, seed=DEFAULT_SEED)[0] for p in (8, 12, 30)}
    for p in (12, 30):
        for col in ["y", "w1", "w2"] + [f"x{j}" for j in range(1, 9)]:
            assert np.array_equal(dfs[8][col].to_numpy(), dfs[p][col].to_numpy()), col


def test_true_beta_nesting():
    b8, b30 = true_beta(8), true_beta(30)
    assert np.array_equal(b8, b30[:8])
    assert np.all(b30[5:] == 0.0)


def test_frozen_files_match_generator(frozen_paths):
    """The committed frozen artifacts are exactly reproducible from the seed."""
    df_new, _ = generate_panel(100, 10, 8, seed=DEFAULT_SEED)
    frozen = pd.read_parquet(frozen_paths[8]["parquet"])
    for col in df_new.columns:
        assert np.array_equal(
            df_new[col].to_numpy(), frozen[col].to_numpy()
        ), f"frozen panel_8 column {col} differs from regeneration"
