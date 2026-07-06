import torch

from src.evaluation import evaluate_retrieval, run_benchmark


def test_evaluate_retrieval_hit_at_1():
    m = evaluate_retrieval([3, 9, 4], [3, 2, 1], k=1)
    assert m == {"Recall@1": 1, "Precision@1": 1.0}


def test_evaluate_retrieval_miss():
    m = evaluate_retrieval([9, 8], [1, 2, 3], k=2)
    assert m == {"Recall@2": 0, "Precision@2": 0.0}


def test_evaluate_retrieval_partial_precision():
    m = evaluate_retrieval([1, 2, 9, 8, 7], [1, 2, 3], k=5)
    assert m["Recall@5"] == 1
    assert m["Precision@5"] == 2 / 5


def test_run_benchmark_perfect_toy_setup():
    # 4 images in a 4-d space; image i = one-hot(i).
    image_features = torch.eye(4)

    # Fake text encoder: "a photo of a smiling person" -> direction of image 1.
    def encode_texts(prompts):
        assert prompts == ["a photo of a smiling person"]
        return torch.tensor([[0.0, 10.0, 0.0, 0.0]])

    # Source image 0 with query +Smiling; correct target is image 1
    # (v_ref + strong text vector points near one-hot(1)).
    annotations = [{"query": "+Smiling", "ground_truth": {"0": [1]}}]

    df = run_benchmark(annotations, image_features, encode_texts)
    row = df.loc[df["query"] == "+Smiling"].iloc[0]
    assert row["R@1"] == 1.0
    assert row["P@1"] == 1.0
    assert row["sources"] == 1
    assert "MEAN" in df["query"].values
