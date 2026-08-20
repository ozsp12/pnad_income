from data import DEFAULT_METADATA_PATH, DEFAULT_REFINED_PATH, DEFAULT_TRUSTED_PATH, load_metadata


def test_default_data_paths_use_data_layers():
    assert DEFAULT_METADATA_PATH.parent.name == "metadata"
    assert DEFAULT_METADATA_PATH.parent.parent.name == "data"
    assert DEFAULT_METADATA_PATH.name == "pnad_metadata.csv"
    assert DEFAULT_REFINED_PATH.name == "refined"
    assert DEFAULT_REFINED_PATH.parent.name == "data"
    assert DEFAULT_TRUSTED_PATH.name == "trusted"
    assert DEFAULT_TRUSTED_PATH.parent.name == "data"


def test_default_metadata_loads_expected_years():
    metadata = load_metadata()
    assert metadata["year"].min() == 1976
    assert metadata["year"].max() == 2025
    assert metadata["year"].is_unique
