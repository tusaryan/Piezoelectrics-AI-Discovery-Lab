from pathlib import Path

from piezo_ml.pipeline import PreprocessingPipeline


def test_pipeline_parses_sample_knn_basic(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[3]
    pipeline = PreprocessingPipeline()
    output = pipeline.run(
        dataset_id="test-dataset",
        csv_path=repo_root / "resources/sample-and-test-dataset/sample_knn_basic.csv",
        required_targets=["d33", "tc"],
    )

    assert len(output.parsed_compositions) == len(output.cleaned)
    assert output.artifact.source_with_uid_path.exists()
    assert output.artifact.parsed_compositions_path.exists()
    assert set(output.parsed_compositions["parse_status"].unique()) == {"success"}
    assert output.parsed_compositions["uid"].tolist() == output.cleaned["uid"].tolist()
