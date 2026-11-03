import pytest
import torch
import numpy as np


def test_skipped_is_not_effective_and_diagnostic_proposals_stay_diagnostic():
    from rba.analysis.hypotheses import _status, rank_hypotheses
    assert _status({'skipped':'tied','mean_delta':None}) == ('未执行',None,None)
    summary = {'key':'1->0','label':'1','pred':'0'}
    mech = {'components':[],'grade_mass':1.,'attribution_reliable':True}
    key = 'lm_head:0->1'
    result = rank_hypotheses(summary,mech,{'tied_lm_head':True},[{'key':key,'skipped':'tied'}],10)[0]
    assert result['status']=='未执行' and result['patch'] is None and result['diagnostic_only']
    result = rank_hypotheses(summary,mech,None,[{'key':key,'n':8,'mean_delta':-2.,'agreement':1.}],10)[0]
    assert result['status']=='仅相关' and result['diagnostic_only']


def test_proposal_scope_is_explicit_and_ablation_does_not_verify_half_scale():
    from rba.analysis.hypotheses import rank_hypotheses
    c = {'key':'L0.self_attn.h0','kind':'attn_head','layer':0,'head':0,'mean_attr':2.,'std_attr':0.,'share':1.}
    v = {'key':c['key'],'n':4,'mean_delta':-2.,'agreement':1.,'scope':'decision',
         'patch':{'method':'head_scale','layer':0,'heads':[0],'scale':0.}}
    result = rank_hypotheses({'key':'1->0','label':'1','pred':'0'},{'components':[c],'attribution_reliable':True},None,[v],10)[0]
    assert result['patch']['scope']=='decision'
    assert result['diagnostic_patch']['scale']==0. and result['patch']['scale']==.5
    assert not result['proposal_tested'] and '未按原样复核' in result['evidence']


def test_display_fallback_and_probe_contract():
    from rba.analysis.delta_w import _topk_scores
    from rba.analysis.token_attr import _top_tokens
    from rba.analysis.probes import _classify_split
    class Tokenizer:
        def convert_ids_to_tokens(self,i): return 'raw-byte'
        def decode(self,ids,**kw): return '\ufffd' if len(ids)==1 else '中文片段'
    tok = Tokenizer()
    entries = _topk_scores(torch.tensor([1.]),tok)
    assert entries[0]['display']=='raw-byte' and entries[0]['id']==0
    entries = _top_tokens(torch.tensor([1.,2.]),torch.tensor([True,True]),torch.tensor([0,1]),tok,1)
    assert entries[0]['display']=='raw-byte' and entries[0]['context']=='中文片段'
    result = _classify_split(np.zeros((12,2)), np.array([0,0]+[1]*10))
    assert result['majority_baseline']==result['majority_accuracy_heldout']


def test_conclusion_reports_counts_without_acceptance_rule():
    from rba.repair.evaluate import conclusion
    text = conclusion({'fix_rate':.5,'net_correct_change':2},{'regression_rate':.1,'net_correct_change':-1})
    assert '+2' in text and '-1' in text and '不能直接比较' in text
    assert '不要保留' not in text


def test_identity_errors_cannot_cancel_in_the_mean():
    from rba.analysis.hypotheses import rank_hypotheses
    c = {'key':'L0.self_attn.h0','kind':'attn_head','layer':0,'head':0,'mean_attr':2.,'std_attr':0.,'share':1.}
    v = [{'key':c['key'],'n':4,'mean_delta':-2.,'agreement':1.},
         {'key':c['key']+':identity','control_type':'identity','target_key':c['key'],
          'mean_delta':0.,'per_case':[{'delta':1.},{'delta':-1.}]}]
    result = rank_hypotheses({'key':'1->0','label':'1','pred':'0'},{'components':[c],'attribution_reliable':True},None,v,10)[0]
    assert result['status']=='证据不足' and result['patch'] is None
    assert f"verification:{c['key']}:identity" in result['evidence_ids']


def test_insufficient_reasons_and_status_rules_are_exported():
    from rba.analysis.hypotheses import rank_hypotheses, STATUS_RULES
    c = {'key':'L0.self_attn.h0','kind':'attn_head','layer':0,'head':0,'mean_attr':2.,'std_attr':0.,'share':1.}
    result = rank_hypotheses({'key':'1->0','label':'1','pred':'0'}, {'components':[c],'attribution_reliable':False},
                            None,[{'key':c['key'],'n':2,'mean_delta':-2.,'agreement':1.}],10)
    target = next(h for h in result if h['key']==c['key'])
    assert '2 条 < 4' in target['insufficient_reason'] and '重建不可靠' in target['insufficient_reason']
    assert target['insufficient_reason'] in target['evidence']
    assert '未执行' in STATUS_RULES and '4' in STATUS_RULES['证据不足']


def test_large_architecture_prompt_keeps_summaries_before_component_details():
    from rba.llm.prompt import build_group_prompt
    components = [{'key':f'L{layer}.self_attn.h{head}','kind':'attn_head','layer':layer,'head':head,
                   'mean_attr':.2,'std_attr':.1,'share':.01} for layer in range(28) for head in range(16)]
    case = {'id':'c','label':'1','pred':'0','head_attr':[[.1]*16 for _ in range(28)],'mlp_attr':[.2]*28,
            'lens':[{'layer':i,'logits':[.1,.2],'probs':[.4,.6]} for i in range(28)]}
    result = build_group_prompt(architecture={'class_name':'Qwen','n_layers':28,'layers':[
        {'index':i,'kind':'full_attention' if i%4==0 else 'linear_attention','heads':16,'head_dim':128} for i in range(28)]},
        grades=['0','1'],grade_ids={'0':1,'1':2},group={'key':'1->0','count':4,'n_detail':4},
        mech={'components':components,'case_evidence':[case],'lens':[{'layer':0,'logit_diff':1.}]},
        hypotheses=[],delta_top=[],template='{query}',budget_tokens=1800,chars_per_token=1.6,
        token_attr={'image_signed':1.,'image_abs_share':.5},delta_report={'lm_head_grades':{'rows':{'0':{'delta_norm':1.}}}})
    visible = result['prompt_evidence_ids']
    assert {'token_attribution','lens','delta','lm_head_grades'} <= set(visible)
    assert 'layer_kind_counts' in result['text']
    assert 'insufficient_evidence' in result['text'] and result['estimated_tokens'] <= 1800
    compact = next(o['data'] for o in result['evidence_packet']['observations'] if o['evidence_id']=='case:c')
    assert len(compact['top_readout_components']) == 8 and compact['omitted_component_count']==468


def test_custom_readout_path_and_checkpoint_scan_count(monkeypatch):
    import copy
    from rba.analysis import delta_w
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8,4)
            self.output_projection = torch.nn.Linear(4,8,bias=False)
            self.output_projection.weight = self.embed.weight
    base = Model()
    ft = copy.deepcopy(base)
    with torch.no_grad(): ft.output_projection.weight[0].add_(.1)
    count = 0
    original = delta_w._iter_pairs
    def counted(*args,**kwargs):
        nonlocal count
        count += 1
        yield from original(*args,**kwargs)
    monkeypatch.setattr(delta_w,'_iter_pairs',counted)
    result = delta_w.analyze_delta(base,ft,None,ft.output_projection.weight,{'0':0,'1':1},None,1,2,16)
    assert result['tied_lm_head'] and result['readout_name']=='output_projection.weight'
    assert result['lm_head_grades']['rows']['0']['delta_norm'] > 0
    assert result['top'][0]['svd']['directions'][0]['space']=='vocab'
    assert count==2


def test_budget_never_keeps_interpretation_without_its_evidence():
    from rba.llm.prompt import fit_budget
    parts = [{'name':'header','text':'H','droppable':False},
             {'name':'evidence','text':'x'*1000,'droppable':True},
             {'name':'claim','text':'SHORT_CLAIM','droppable':True,'requires':['evidence']},
             {'name':'ask','text':'JSON','droppable':False}]
    text,dropped = fit_budget(parts,50,1)
    assert 'SHORT_CLAIM' not in text and set(dropped)=={'evidence','claim'}


def test_unknown_vocab_alignment_cannot_supply_readout_semantics():
    from rba.llm.prompt import _compact_delta
    from rba.analysis.hypotheses import propose_checks
    delta = {'base_grade_alignment':{'status':'unavailable'},'lm_head_grades':{'rows':{'0':{'delta_norm':10.}}}}
    checks = propose_checks({'key':'1->0','pred':'0','label':'1'},{'components':[]},delta,1)
    assert not any(c['patch']['method']=='lm_head_blend' for c in checks)
    rows = [{'name':'lm_head.weight','svd':{'directions':[{'space':'vocab','grades':{'0':1.},'top_tokens':[{'token':'zero'}]}]}}]
    compact = _compact_delta(rows, alignment_known=False)
    direction = compact[0]['svd']['directions'][0]
    assert 'grades' not in direction and 'top_tokens' not in direction
    assert direction['semantic_projection_unavailable']
    assert 'grades' in rows[0]['svd']['directions'][0]  # audit source not mutated
