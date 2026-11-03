import pytest
import torch
from types import SimpleNamespace

from rba.errors import RBAError
from rba.models.loader import resolve_grade_ids
from rba.repair.evaluate import forward_scores, score_rows, compare, conclusion, wilson_interval


class GradeTokenizer:
    def __len__(self): return 10
    def encode(self, text, **kw): return {'-1': [1, 2], '0': [3], '1': [2]}[text]
    def decode(self, ids, **kw): return {(1,2): '-1', (3,): '0', (2,): '1'}[tuple(ids)]


def test_grade_contract_integer_keys_and_declared_prefix():
    t = GradeTokenizer()
    assert resolve_grade_ids(t, ['-1','0'], {-1:1, 0:3}, 'first_token') == {'-1':1, '0':3}
    with pytest.raises(RBAError): resolve_grade_ids(t, ['-1','0'], {-1:1,0:3})
    for mapping in ({'-1':3,'0':3}, {'-1':10,'0':3}, {'-1':1,'0':True}, {-1:1,'-1':1,'0':3}):
        with pytest.raises(RBAError): resolve_grade_ids(t, ['-1','0'], mapping, 'first_token')
    t.encode = lambda text, **kw: [1,2] if text == '-1' else [1]
    t.decode = None
    with pytest.raises(RBAError): resolve_grade_ids(t, ['-1','0'], None, 'first_token')


class ConstantModel(torch.nn.Module):
    def forward(self, input_ids, **kw):
        return SimpleNamespace(logits=torch.tensor([0., 2., 1., 4.]).expand(*input_ids.shape, 4))


def test_free_decode_does_not_count_illegal_grade_as_correct():
    model = ConstantModel()
    batch = {'input_ids':torch.ones(1,2,dtype=torch.long),'attention_mask':torch.ones(1,2,dtype=torch.long)}
    scores = forward_scores(model,batch,[1,2],torch.device('cpu'),torch.float32)
    assert score_rows(scores,['0','1'],'constrained')[0]['pred'] == '0'
    row = score_rows(scores,['0','1'],'free')[0]
    assert row['pred'] == '__invalid__' and not row['legal']
    assert 0 < row['grade_mass'] < .5
    assert not hasattr(model,'_rba_input_ids')


def test_paired_statistics_and_undefined_controls():
    before = [{'id':'a','label':'0','pred':'1','legal':True,'attributes':{'domain':'a'}},
              {'id':'b','label':'1','pred':'1','legal':True,'attributes':{'domain':'b'}}]
    after = [{'id':'a','label':'0','pred':'0','legal':True}, {'id':'b','label':'1','pred':'__invalid__','legal':False}]
    result = compare(before,after)
    assert result['net_correct_change'] == 0 and result['n_new_illegal'] == 1
    assert result['per_attribute']['domain']['b']['regression_rate'] == 1
    assert result['fix_rate_ci'][0] < 1
    assert '非法' in conclusion(result,result)
    assert '没有原本判对' in conclusion(result,compare(before[:1],after[:1]))
    assert wilson_interval(0,0) is None
    with pytest.raises(RBAError): compare(before,after[:1])


def _tiny_case(dtype=torch.float32):
    from rba.config import Config
    from rba.models.loader import load_models
    from rba.models.paths import resolve_paths
    from rba.data.encode import encode_cases, collate
    from rba.schema import Case
    cfg = Config(device='cpu', dtype='float32', seed=0, plant_error=True)
    loaded = load_models(cfg)
    loaded.ft.to(dtype=dtype)
    resolved = resolve_paths(loaded.ft, {}, 'qwen3_5')
    with torch.no_grad():
        norm = resolved.final_norm(loaded.ft)
        norm.weight.copy_(torch.linspace(-.8, 2., norm.weight.numel()))
    case = Case(id='a',query='天气',doc='欧冠决赛',image=None,label='3')
    batch = collate(encode_cases([case],loaded.tokenizer,cfg),0)
    return cfg, loaded, resolved, batch


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_norm_aware_dla_reconstructs_with_nonuniform_zero_centered_norm(dtype):
    from rba.analysis.instrument import instrument_batch
    from rba.analysis.hypotheses import aggregate_mechanistic
    cfg, loaded, resolved, batch = _tiny_case(dtype)
    ids = [loaded.grade_ids[g] for g in cfg.grades]
    record = instrument_batch(loaded.ft,batch,resolved,ids,torch.tensor([ids[1]]),torch.tensor([ids[-1]]),
                              loaded.image_token_id,loaded.device,dtype)[0]
    assert record['attribution_reliable']
    assert record['norm_component_sum'] == pytest.approx(record['logit_diff'], rel=.05, abs=.05)
    assert abs(record['prenorm_dot'] - record['logit_diff']) > .1
    assert all('logits' in x for x in record['lens'])
    for heads, attn in zip(record['norm_head_attr'],record['norm_attn_attr']):
        if heads is not None:
            assert sum(heads) == pytest.approx(attn,rel=.03,abs=.03)
    record.update(label='3',pred='0')
    summary = aggregate_mechanistic([record],cfg.grades)
    assert summary['readout_kind'] == 'final_norm_dla'
    assert summary['attribution_reliable']
    assert summary['case_evidence'][0]['id']=='a'
    assert summary['case_evidence'][0]['lens'][-1]['logits'] == record['lens'][-1]['logits']


def test_norm_direction_handles_standard_norm_and_sign_reversal():
    from rba.analysis.instrument import norm_readout_direction
    norm = torch.nn.RMSNorm(2,eps=1e-6)
    with torch.no_grad(): norm.weight.copy_(torch.tensor([.1,10.]))
    r = torch.tensor([[1.,1.]])
    d = torch.tensor([[1.,1.]])
    c = torch.tensor([[2.,-1.]])
    actual = norm_readout_direction(norm,r,d)
    assert (c*d).sum() > 0 and (c*actual).sum() < 0
    assert norm_readout_direction(torch.nn.LayerNorm(2),r,d) is None


def test_svd_factors_activation_conditioned_write_and_tied_rows():
    from rba.analysis.delta_w import analyze_delta, _svd_project
    from rba.analysis.instrument import instrument_batch
    cfg, loaded, resolved, batch = _tiny_case()
    matrix = torch.tensor([[2.,-1.],[0.,3.]])
    factors = _svd_project('x.o_proj.weight',matrix,torch.eye(2),{'0':0,'1':1},None,2,16,2)
    reconstruction = sum(d['singular_value'] * torch.outer(torch.tensor(d['left_vector']),torch.tensor(d['right_vector'])) for d in factors['directions'])
    assert torch.allclose(matrix,reconstruction,atol=1e-5)
    assert factors['explained_energy'] == pytest.approx(1.,abs=1e-6)
    delta = analyze_delta(loaded.base,loaded.ft,None,loaded.ft.lm_head.weight,loaded.grade_ids,loaded.tokenizer,8,2,256)
    ids = [loaded.grade_ids[g] for g in cfg.grades]
    record = instrument_batch(loaded.ft,batch,resolved,ids,torch.tensor([ids[1]]),torch.tensor([ids[-1]]),
                              loaded.image_token_id,loaded.device,torch.float32,delta=delta)[0]
    assert record['delta_contributions']
    assert any('readout_logit_diff' in d for d in record['delta_contributions'])
    # Physical alias must be reported and each parameter counted only once.
    for model in (loaded.base,loaded.ft):
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
    tied = analyze_delta(loaded.base,loaded.ft,None,loaded.ft.lm_head.weight,loaded.grade_ids,loaded.tokenizer,2,1,256)
    assert tied['tied_lm_head']
    assert tied['lm_head_grades']['rows']['0']['cosine_to_base'] is not None
    assert tied['n_parameters_compared'] == sum(p.numel() for p in loaded.ft.parameters())


def test_hypotheses_controls_negative_evidence_and_image_counterexample():
    from rba.analysis.hypotheses import propose_checks, rank_hypotheses
    summary = {'key':'3->0','label':'3','pred':'0','has_image_fraction':1.}
    components = [{'key':f'L0.self_attn.h{i}','kind':'attn_head','layer':0,'head':i,'mean_attr':1.,'std_attr':0.,'share':.5} for i in range(2)]
    mech = {'components':components,'attribution_reliable':True,'grade_mass':.9,'image_token_fraction':.99}
    checks = propose_checks(summary,mech,None,1)
    assert [c['control_type'] for c in checks] == ['target','random','identity']
    assert checks[1]['patch']['heads'] == [1] and checks[2]['patch']['scale'] == 1
    assert checks == propose_checks(summary,mech,None,1)
    ranked = rank_hypotheses(summary,mech,None,[],100,token_attr={'image_signed':0.,'image_abs_share':0.})
    assert not any(h['key']=='image_tokens' for h in ranked)
    assert all(h['confidence'] is None for h in ranked)
    bad_identity = [{'key':components[0]['key'],'n':4,'mean_delta':-1.,'agreement':1.},
                    {'key':components[0]['key']+':identity','target_key':components[0]['key'],'control_type':'identity','mean_delta':.2}]
    ranked = rank_hypotheses(summary,mech,None,bad_identity,100)
    h = next(h for h in ranked if h['key']==components[0]['key'])
    assert h['status']=='证据不足' and h['patch'] is None


def test_probe_intercept_uses_majority_when_no_feature_signal():
    import numpy as np
    from rba.analysis.probes import _classify_split
    # Old centered predictions omitted intercept, picked class 0 for all ties.
    result = _classify_split(np.zeros((12,2)),np.array([0,0]+[1]*10))
    assert result['accuracy_train'] == pytest.approx(10/12)
    assert result['majority_accuracy_heldout'] is not None
    assert result['n_train']==6 and result['n_test']==6


def test_prompt_packet_keeps_controls_and_schema_without_confidence():
    from rba.llm.prompt import build_group_prompt, fit_budget
    result = build_group_prompt(architecture={'class_name':'tiny'},grades=['0','1'],grade_ids={'0':1,'1':2},
        group={'key':'1->0','label':'1','pred':'0','count':4,'n_detail':2,'examples':[]},
        mech={'components':[],'lens':[],'grade_mass':.8}, hypotheses=[],delta_top=[],template='{query}',
        budget_tokens=4000,chars_per_token=1.6,verifications=[{'key':'a','control_type':'identity','mean_delta':0.,'per_case':[]}],
        grade_mode='first_token',decode_protocol='free',delta_report={'tied_lm_head':True,'lm_head_grades':{'rows':{'0':{'delta_norm':1.,'delta_vector':[1.,2.]}}}})
    packet = result['evidence_packet']
    assert packet['schema_version']==1 and packet['protocol']['grade_mode']=='first_token'
    head = next(o['data'] for o in packet['observations'] if o['evidence_id']=='lm_head_grades')
    assert head['tied_lm_head'] and head['rows']['0']['delta_norm']==1.
    assert 'delta_vector' not in head['rows']['0']
    assert any(o['evidence_id']=='verification:a' for o in packet['observations'])
    assert 'insufficient_evidence' in result['text']
    parts = [{'name':'header','text':'HEAD','droppable':False}, {'name':'large','text':'x'*1000,'droppable':True}, {'name':'ask','text':'JSON_END','droppable':False}]
    text,dropped = fit_budget(parts,20,1)
    assert text.endswith('JSON_END') and 'large' in dropped
    with pytest.raises(RBAError): fit_budget(parts,2,1)


def test_tied_checkpoint_alias_and_untied_base_head_are_not_confused(tmp_path):
    from safetensors.torch import save_file
    from rba.analysis.delta_w import analyze_delta, _iter_pairs
    cfg, loaded, resolved, _ = _tiny_case()
    loaded.ft.lm_head.weight = loaded.ft.model.language_model.embed_tokens.weight
    expected = loaded.base.lm_head.weight.detach().clone()
    pairs = {name:(base,ft) for name,base,ft in _iter_pairs(loaded.base,loaded.ft,None)}
    assert torch.equal(pairs['lm_head.weight'][0],expected)
    loaded.base.lm_head.weight = loaded.base.model.language_model.embed_tokens.weight
    state = {name:p.detach().clone() for name,p in loaded.base.state_dict().items() if name != 'lm_head.weight'}
    save_file(state,str(tmp_path/'model.safetensors'))
    (tmp_path/'config.json').write_text('{"tie_word_embeddings":true}')
    report = analyze_delta(None,loaded.ft,str(tmp_path),loaded.ft.lm_head.weight,loaded.grade_ids,loaded.tokenizer,1,1,256)
    assert report['tied_lm_head'] and report['n_missing']==0
    assert report['lm_head_grades']['base_source'].endswith('embed_tokens.weight')
    assert report['lm_head_grades']['rows']['0']['delta_norm'] >= 0


def test_readout_bias_is_included_in_lens_and_reconstruction():
    from rba.analysis.instrument import instrument_batch
    cfg, loaded, resolved, batch = _tiny_case()
    old = loaded.ft.lm_head
    replacement = torch.nn.Linear(old.in_features,old.out_features,bias=True)
    with torch.no_grad():
        replacement.weight.copy_(old.weight)
        replacement.bias.copy_(torch.linspace(-1.,1.,old.out_features))
    loaded.ft.lm_head = replacement
    ids = [loaded.grade_ids[g] for g in cfg.grades]
    record = instrument_batch(loaded.ft,batch,resolved,ids,torch.tensor([ids[1]]),torch.tensor([ids[-1]]),
                              loaded.image_token_id,loaded.device,torch.float32)[0]
    assert record['readout_bias_diff'] != 0
    assert record['attribution_reliable']
    assert record['lens'][-1]['probs'] == pytest.approx(record['cond_probs'],abs=1e-5)


def test_real_qwen_nonuniform_final_norm_readout():
    pytest.importorskip('transformers.models.qwen3_5')
    from rba.models.qwen_tiny import build_random_qwen35
    from rba.models.paths import resolve_paths
    from rba.analysis.instrument import instrument_batch
    model = build_random_qwen35(seed=13)
    resolved = resolve_paths(model,{},'qwen3_5')
    norm = resolved.final_norm(model)
    with torch.no_grad(): norm.weight.copy_(torch.linspace(-.5,1.5,norm.weight.numel()))
    ids = torch.tensor([[1,4,5,6,7]])
    row = instrument_batch(model,{'input_ids':ids,'attention_mask':torch.ones_like(ids),'case_ids':['q']},
                            resolved,[2,3,4,5,6],torch.tensor([4]),torch.tensor([2]),10,torch.device('cpu'),torch.float32)[0]
    assert row['attribution_reliable']
    assert row['norm_component_sum'] == pytest.approx(row['logit_diff'],abs=1e-4)
