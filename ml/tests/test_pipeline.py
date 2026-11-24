import numpy as np

from ml.color import color_compat, dominant_colors, nearest_named_color
from ml.data.synthetic import make_eval, make_synthetic
from ml.models.compat import load_checkpoint, save_checkpoint
from ml.scorers import ColorRuleScorer, CosineScorer, ModelScorer, RandomScorer, compat_auc, fitb_accuracy
from ml.train import train


def test_train_beats_baselines_on_synthetic(tmp_path):
    store, outfits = make_synthetic(n_outfits=1500)
    fitb, compat = make_eval(store, outfits, n=300)
    model, info = train(store, outfits[:1200], fitb, compat, epochs=8, log=lambda *_: None)
    m_acc = fitb_accuracy(ModelScorer(model), store, fitb)[0]
    c_acc = fitb_accuracy(CosineScorer(), store, fitb)[0]
    r_acc = fitb_accuracy(RandomScorer(), store, fitb)[0]
    assert m_acc > 0.5 and m_acc > c_acc and m_acc > r_acc
    assert compat_auc(ModelScorer(model), store, compat)[0] > 0.7
    assert np.isfinite(fitb_accuracy(ColorRuleScorer(), store, fitb)[0])

    path = str(tmp_path / "c.pt")
    save_checkpoint(model, path, info)
    loaded, meta = load_checkpoint(path)
    x = store.emb[:4], store.types[:4]
    assert np.allclose(model.score_numpy(x[0], x[1], x[0][::-1], x[1][::-1]), loaded.score_numpy(x[0], x[1], x[0][::-1], x[1][::-1]), atol=1e-5)


def test_color_rules():
    assert nearest_named_color((25, 40, 90)) == "navy"
    assert color_compat((10, 10, 10), (200, 30, 40)) == 0.8              # neutral matches anything
    assert color_compat((200, 30, 40), (30, 200, 190)) > color_compat((200, 30, 40), (200, 140, 30)) - 0.5
    px = np.concatenate([np.full((500, 3), 255), np.tile([27, 45, 95], (200, 1))]).astype(np.uint8)
    assert nearest_named_color(dominant_colors(px)[0][0]) == "navy"      # white background ignored


def test_accessories_do_not_disqualify_questions():
    """Polyvore outfits nearly always contain a bag or jewellery that is never embedded; those items must be
    dropped from the question, not cause the whole question / outfit to be skipped."""
    store, outfits = make_synthetic(n_outfits=200)
    fitb, compat = make_eval(store, outfits, n=50)
    for q in fitb:
        q["question"] = q["question"] + ["some_unembedded_bag"]
    compat = [(label, ids + ["some_unembedded_bag"]) for label, ids in compat]
    assert fitb_accuracy(CosineScorer(), store, fitb)[1] == 50
    assert compat_auc(CosineScorer(), store, compat)[1] == 100
    fitb[0]["answers"][2] = "accessory_answer"          # a blank that is an accessory: skip that question
    assert fitb_accuracy(CosineScorer(), store, fitb)[1] == 49


def test_ties_and_answer_position_do_not_inflate_fitb(tmp_path):
    """A scorer that gives every pair the same score must land on chance (0.25), wherever the true answer sits."""
    import json

    from ml.data.polyvore import load_fitb
    from ml.scorers import Scorer

    class Constant(Scorer):
        def pair_scores(self, store, ra, rb):
            return np.full(len(ra), 0.5)

    store, outfits = make_synthetic(n_outfits=200)
    fitb, _ = make_eval(store, outfits, n=100)
    assert len({q["correct"] for q in fitb}) > 1
    assert abs(fitb_accuracy(Constant(), store, fitb)[0] - 0.25) < 1e-9

    # loader: the true answer is the token from the question's own set, at any position
    raw = [{"question": ["77_1", "77_2"], "blank_position": 3, "answers": ["12_4", "90_1", "77_3", "33_2"]},
           {"question": ["5_1"], "blank_position": 2, "answers": ["8_1", "9_1", "10_1", "11_1"]}]
    path = tmp_path / "fitb.json"
    path.write_text(json.dumps(raw))
    loaded = load_fitb(str(path), {"77_3": "itemX"})
    assert len(loaded) == 1 and loaded[0]["correct"] == 2 and loaded[0]["answers"][2] == "itemX"
