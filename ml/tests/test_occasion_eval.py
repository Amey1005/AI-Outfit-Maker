import numpy as np

from ml.occasion_eval import evaluate


def test_occasion_eval_prompt_and_logreg():
    rng = np.random.default_rng(0)
    centers = np.eye(4) * 3
    y = np.repeat(np.arange(4), 20)
    x = centers[y] + rng.normal(size=(80, 4)) * 0.5
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    labels = ["college", "party", "formal", "gym"]
    names = [labels[i] for i in y]
    r = evaluate(x, names, np.eye(4))
    assert r["accuracy"] > 0.9 and abs(r["majority_baseline"] - 0.25) < 1e-9
    assert evaluate(x, names, np.eye(4), logreg=True)["accuracy"] > 0.9
