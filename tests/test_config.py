from pnad_income.data import DEFAULT_METADATA_PATH, load_metadata


def test_default_metadata_path_uses_metadata_directory():
    assert DEFAULT_METADATA_PATH.parent.name == "metadata"
    assert DEFAULT_METADATA_PATH.name == "pnad_metadata.csv"


def test_default_metadata_loads_expected_years():
    metadata = load_metadata()
    assert metadata["year"].min() == 1976
    assert metadata["year"].max() == 2025
    assert metadata["year"].is_unique
