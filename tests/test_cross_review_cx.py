"""Regression checks for engineering cross-review of fa1511c."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import rba.pipeline as pipeline
from rba.config import Config
from rba.errors import RBAError
from rba.models.loader import is_tied_lm_head
from rba.schema import Case


def test_duplicate_content_cannot_cross_holdout():
    cases = [Case(str(i), 'same query', 'same doc', None, '0') for i in range(20)]
    decisions = {int.from_bytes(hashlib.sha256(f'0:{c.id}'.encode()).digest()[:8], 'big') / 2**64 < .5 for c in cases}
    assert decisions == {True, False}
    with pytest.raises(RBAError, match='验收集不独立'):
        pipeline._split_info(cases, Config(holdout_fraction=.5, seed=0))
    assert pipeline._split_info(cases, Config(holdout_fraction=0))['n_holdout'] == 0


def test_control_ids_cannot_overwrite_badcase_records(tmp_path):
    paths = []
    for name, label in [('bad', 0), ('good', 3)]:
        path = tmp_path / f'{name}.jsonl'
        path.write_text(json.dumps({'id': 'same', 'query': name, 'doc': name, 'label': label}) + '\n')
        paths.append(str(path))
    with pytest.raises(RBAError, match='id 重复'):
        pipeline._validate_and_load(Config(badcases=paths[0], goodcases=paths[1]))


@pytest.mark.parametrize('swapped', [False, True])
def test_grade_ids_alone_do_not_prove_vocab_alignment(monkeypatch, swapped):
    class Tokenizer:
        def __init__(self, vocab):
            self.vocab = vocab
        def encode(self, text, **kwargs):
            return [self.vocab[text]]
        def get_vocab(self):
            return self.vocab
    ft = Tokenizer({'0': 0, 'cat': 1, 'dog': 2})
    base = Tokenizer({'0': 0, 'cat': 2, 'dog': 1} if swapped else dict(ft.vocab))
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: base)
    loaded = SimpleNamespace(kind='hf', base_path='base', tokenizer=ft)
    status = pipeline.check_base_grade_alignment(Config(), loaded, {'0': 0})
    assert status['status'] == ('unavailable' if swapped else 'aligned')


def test_custom_tied_head_blocks_readout_only_patch(monkeypatch):
    embed = torch.nn.Embedding(8, 4)
    custom = torch.nn.Linear(4, 8, bias=False)
    custom.weight = embed.weight
    model = SimpleNamespace(lm_head=torch.nn.Linear(4, 8, bias=False), custom=custom,
                            get_input_embeddings=lambda: embed)
    assert not is_tied_lm_head(model)
    assert is_tied_lm_head(model, custom)
    loaded = SimpleNamespace(ft=model, preset='test', grades=['0'], grade_ids={'0': 0}, tied_lm_head=False)
    monkeypatch.setattr(pipeline, '_validate_and_load', lambda cfg: ([], [], {}))
    monkeypatch.setattr(pipeline, 'load_models', lambda cfg: loaded)
    monkeypatch.setattr(pipeline, 'resolve_paths', lambda *args: SimpleNamespace(lm_head=lambda m: m.custom))
    with pytest.raises(RBAError, match='共享权重'):
        pipeline.evaluate_patch(Config(grades=['0']), {'method': 'lm_head_blend'})
