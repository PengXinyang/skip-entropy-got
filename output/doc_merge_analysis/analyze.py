"""Read-only analysis of saved GoT runs; no model calls. Figures use English labels."""
import argparse
import collections
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

STAGES = {1: 'Pair A', 5: 'Pair B', 8: 'Aggregate', 11: 'Refine'}
COLORS = ['#64748b', '#168a87', '#db7842']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), q25=float(np.quantile(a, .25)),
                median=float(np.median(a)), q75=float(np.quantile(a, .75)), max=float(a.max()))


def tagged_number(text, tag):
    m = re.search(r'<' + tag + r'>\s*(\d+(?:\.\d+)?)\s*</' + tag + r'>', text or '')
    return float(m[1]) if m else None


def origin(metadata, by_id):
    op = by_id.get(metadata.get('operation_id'), {})
    return (op.get('operation_index'), metadata.get('thought_index'))


def inspect_graph(graph, case_id, variant):
    ops = [o for o in graph if 'operation' in o]
    by_id = {o['operation_id']: o for o in ops}
    scored = {}
    selected = set()
    for op in ops:
        for j, metadata in enumerate(op.get('thought_metadata', [])):
            key = origin(metadata, by_id)
            if op['operation'] == 'score':
                scored[key] = (op['scores'][j], metadata.get('score_observation', {}))
            elif op['operation'] == 'keep_best_n':
                selected.add(key)
    final = ops[-1]
    final_state = final['thoughts'][0]
    final_meta = final['thought_metadata'][0]
    final_origin = origin(final_meta, by_id)
    nodes = []
    for op in ops:
        if op['operation'] not in ('generate', 'aggregate', 'improve'):
            continue
        for j, (state, meta) in enumerate(zip(op['thoughts'], op['thought_metadata'])):
            key = (op['operation_index'], j)
            score, observation = scored.get(key, (None, {}))
            response = meta.get('response_text', '') or ''
            text = state.get('current', '') or ''
            skipped = bool(meta.get('skipped'))
            obs_text = observation.get('response_text', '') or ''
            nodes.append(dict(case_id=case_id, variant=variant, operation_index=key[0],
                thought_index=j, operation=op['operation'], stage=STAGES.get(key[0], str(key[0])),
                skipped=skipped, entropy=meta.get('normalized_avg_entropy_bits'),
                raw_entropy=meta.get('avg_entropy_bits'), sum_entropy=meta.get('sum_entropy_bits'),
                finish_length=meta.get('finish_reason') == 'length', empty_response=not response.strip(),
                empty_current=not text.strip(), complete_merged_tag='<Merged>' in response and '</Merged>' in response,
                output_chars=len(text), prompt_tokens=meta.get('usage', {}).get('prompt_tokens', 0) or 0,
                completion_tokens=meta.get('usage', {}).get('completion_tokens', 0) or 0,
                score=score, selected_by_keepbest=key in selected, is_final_source=key == final_origin,
                score_call_skipped=bool(observation.get('skipped_score_call')),
                recorded_score_finish_length=observation.get('finish_reason') == 'length',
                recorded_redundancy=tagged_number(obs_text, 'Redundancy'),
                recorded_retained=tagged_number(obs_text, 'Retained')))
    current = final_state.get('current', '') or ''
    return dict(case_id=case_id, variant=variant, **graph[-1], final_score=final.get('scores', [None])[0],
                final_empty=not current.strip(), final_chars=len(current),
                final_source_operation=final_origin[0], final_source_thought=final_origin[1]), nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases, nodes, rankings = [], [], []
    manifests = {}
    for path in args.root.rglob('full_graph.json'):
        case_id = int(path.parent.name.removeprefix('id'))
        if case_id in manifests:
            raise ValueError(f'Duplicate case id {case_id}')
        manifests[case_id] = str(path.parent.resolve())
    if set(manifests) != set(range(100)):
        raise ValueError('Expected exactly ids 0..99')
    for case_id, directory in sorted(manifests.items()):
        base = Path(directory)
        for variant in ('full', 'low', 'high'):
            path = base / ('full_graph.json' if variant == 'full' else f'{variant}/compressed_graph.json')
            case, rows = inspect_graph(read(path), case_id, variant)
            case['cohort'] = 'resumed' if 'resumed_completed' in directory else 'new'
            if variant != 'full':
                summary = read(base / variant / 'summary.json')
                for metric in ('total_tokens', 'api_calls', 'cost', 'total_latency_seconds'):
                    assert abs(case[metric] - summary['compressed'][metric]) < 1e-6, (case_id, variant, metric)
                case['actual_candidate_skip_ratio'] = summary['reductions']['thought_skip_ratio']
                case['num_candidates'] = summary['num_candidates']
                case['num_selected'] = summary['num_skipped_thoughts']
                assert case['num_selected'] == sum(n['skipped'] for n in rows), (case_id, variant, 'skip count')
                case['actual_all_generation_skip_ratio'] = sum(n['skipped'] for n in rows) / len(rows)
                for rank in read(base / variant / 'candidate_ranking.json'):
                    rankings.append(dict(case_id=case_id, variant=variant, **rank))
            cases.append(case)
            nodes.extend(rows)
    indexed = {(c['case_id'], c['variant']): c for c in cases}
    node_index = {(n['case_id'], n['variant'], n['operation_index'], n['thought_index']): n for n in nodes}
    stats = dict(source=str(args.root.resolve()), case_count=100, generation_slot_count=len(nodes), variants={})
    rng = np.random.default_rng(20260907)
    sample = rng.integers(0, 100, size=(10000, 100))
    for variant in ('full', 'low', 'high'):
        rows = [indexed[(i, variant)] for i in range(100)]
        ns = [n for n in nodes if n['variant'] == variant and not n['skipped']]
        vs = dict(final_empty_count=sum(r['final_empty'] for r in rows),
            final_score=describe([r['final_score'] for r in rows]),
            final_score_nonempty=describe([r['final_score'] for r in rows if not r['final_empty']]),
            final_chars=describe([r['final_chars'] for r in rows]), generation_calls=len(ns),
            empty_generation_responses=sum(n['empty_response'] for n in ns),
            truncated_generation_responses=sum(n['finish_length'] for n in ns),
            entropy_available=sum(n['entropy'] is not None for n in ns),
            final_source_counts=dict(collections.Counter(r['final_source_operation'] for r in rows)),
            total={k:sum(r[k] for r in rows) for k in ('total_tokens','prompt_tokens','completion_tokens','api_calls','cost','total_latency_seconds')})
        if variant != 'full':
            full = [indexed[(i, 'full')] for i in range(100)]
            vs['reductions'] = {}
            for metric in ('total_tokens', 'api_calls', 'cost', 'total_latency_seconds'):
                a = np.array([r[metric] for r in full]); b = np.array([r[metric] for r in rows])
                ci = np.quantile(1 - b[sample].sum(axis=1)/a[sample].sum(axis=1), [.025, .975])
                vs['reductions'][metric] = dict(pooled=float(1-b.sum()/a.sum()), bootstrap95=ci.tolist(),
                    per_case=describe(1-b/a), cases_saving=int((b<a).sum()), cases_spending_more=int((b>a).sum()))
            vs['actual_candidate_skip_ratio'] = describe([r['actual_candidate_skip_ratio'] for r in rows])
            vs['actual_all_generation_skip_ratio'] = describe([r['actual_all_generation_skip_ratio'] for r in rows])
            vs['selected_per_case_counts'] = dict(collections.Counter(r['num_selected'] for r in rows))
            vs['zero_skip_ids'] = [r['case_id'] for r in rows if r['num_selected']==0]
            vs['final_zero_score_nonempty'] = sum(not r['final_empty'] and r['final_score']==0 for r in rows)
            vs['transitions'] = {f'{a}_to_{b}':sum(('empty' if f['final_empty'] else 'nonempty') == a and
                ('empty' if r['final_empty'] else 'nonempty') == b for f,r in zip(full,rows))
                for a in ('nonempty','empty') for b in ('nonempty','empty')}
            vs['nonempty_to_empty_ids'] = [i for i in range(100) if not full[i]['final_empty'] and rows[i]['final_empty']]
            selected_ranks = [r for r in rankings if r['variant'] == variant and r['selected_for_skip']]
            selected_nodes = [node_index[(r['case_id'], 'full', r['operation_index'], r['thought_index'])] for r in selected_ranks]
            vs['selected_count'] = len(selected_ranks)
            vs['selected_stage_counts'] = dict(collections.Counter(n['stage'] for n in selected_nodes))
            vs['selected_keepbest_count'] = sum(n['selected_by_keepbest'] for n in selected_nodes)
            vs['selected_final_source_count'] = sum(n['is_final_source'] for n in selected_nodes)
            both = [(f,r) for f,r in zip(full,rows) if not f['final_empty'] and not r['final_empty']]
            vs['both_nonempty_score_delta'] = describe([r['final_score']-f['final_score'] for f,r in both])
            vs['cohort_token_reductions'] = {cohort: 1-sum(r['total_tokens'] for r in rows if r['cohort']==cohort)/sum(r['total_tokens'] for r in full if r['cohort']==cohort) for cohort in ('resumed','new')}
        stats['variants'][variant] = vs
    baseline=np.array([indexed[(i,'full')]['total_tokens'] for i in range(100)])
    low=np.array([indexed[(i,'low')]['total_tokens'] for i in range(100)])
    high=np.array([indexed[(i,'high')]['total_tokens'] for i in range(100)])
    diff=(high[sample].sum(axis=1)-low[sample].sum(axis=1))/baseline[sample].sum(axis=1)
    stats['low_minus_high_token_savings'] = dict(pooled=float((high.sum()-low.sum())/baseline.sum()),
        paired_case_bootstrap95=np.quantile(diff,[.025,.975]).tolist())
    full_nodes = [n for n in nodes if n['variant']=='full']
    observed = [n for n in full_nodes if n['entropy'] is not None]
    stats['full_entropy'] = describe([n['entropy'] for n in observed])
    stats['full_empty_response_token_usage'] = dict(
        total_tokens=sum(n['prompt_tokens']+n['completion_tokens'] for n in full_nodes if n['empty_response']),
        all_generation_tokens=sum(n['prompt_tokens']+n['completion_tokens'] for n in full_nodes))
    stats['full_node_stages'] = {}
    for op, name in STAGES.items():
        ns = [n for n in full_nodes if n['operation_index']==op]
        es = [n for n in ns if n['entropy'] is not None]
        stats['full_node_stages'][name] = dict(slots=len(ns), entropy=describe([n['entropy'] for n in es]),
            empty_responses=sum(n['empty_response'] for n in ns), truncated=sum(n['finish_length'] for n in ns),
            score_zero_nonempty=sum(n['score']==0 and not n['empty_current'] for n in ns))
    def corr(ns):
        if len(ns)<3: return None
        rho=spearmanr([n['entropy'] for n in ns],[n['score'] for n in ns]).statistic
        return dict(n=len(ns), spearman_rho=float(rho) if np.isfinite(rho) else None)
    stats['entropy_score_descriptive_correlations'] = dict(pooled=corr(observed), by_stage={name:corr([n for n in observed if n['stage']==name]) for name in STAGES.values()})
    stats['full_first_score_observation'] = dict(nonempty_scored_nodes=sum(not n['score_call_skipped'] for n in full_nodes),
        with_both_numeric_tags=sum(n['recorded_redundancy'] is not None and n['recorded_retained'] is not None for n in full_nodes),
        truncated=sum(n['recorded_score_finish_length'] for n in full_nodes))
    stats['notes'] = [
        'Node counts use only original Generate/Aggregate records, not copied metadata in Score/KeepBest.',
        'Stored doc_merge score is harmonic mean of averaged Redundancy and Retained; higher Redundancy means less repetition.',
        'Score uses 3 responses but score_observation saves only the first response: component means cannot be exactly recovered.',
        'Bootstrap resamples cases; intervals describe sampling variability conditional on saved runs, not repeated-generation variability.',
        'Correlations are descriptive; nodes are clustered by case and stage. Missing entropy is excluded, not imputed as zero.',
        'Stage indices reflect got2; final_source means retained origin, not full causal ancestry.',
        'Pooled token savings exclude the cost of obtaining the full graph used to select skip positions.',
        'Nonempty output and internal model score are proxies, not verified task correctness.'
    ]
    for filename, data in [('summary.json',stats),('case_metrics.json',cases),('node_metrics.json',nodes),('manifest.json',manifests)]:
        (args.output/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),encoding='utf-8')
    make_figures(cases, nodes, rankings, stats, args.output)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


def make_figures(cases,nodes,rankings,stats,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),bbox_inches='tight',facecolor='white')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    keys=['total_tokens','api_calls','cost','total_latency_seconds']
    for j,v in enumerate(['low','high']):
        rs=stats['variants'][v]['reductions']; y=np.array([100*rs[k]['pooled'] for k in keys]); ci=np.array([rs[k]['bootstrap95'] for k in keys])*100
        axs[0].bar(np.arange(4)+(j-.5)*.36,y,.36,color=COLORS[j+1],label=v.title())
        axs[0].errorbar(np.arange(4)+(j-.5)*.36,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='none',ecolor='#334155',capsize=3)
    axs[0].set_xticks(range(4),['Tokens','API calls','Cost*','API latency']); axs[0].set_ylabel('Reduction vs Full (%)'); axs[0].axhline(0,color='#94a3b8',lw=.8); axs[0].legend(); axs[0].set_title('Pooled savings; case bootstrap 95% CI')
    idx={(c['case_id'],c['variant']):c for c in cases}
    data=[[100*(1-idx[(i,v)]['total_tokens']/idx[(i,'full')]['total_tokens']) for i in range(100)] for v in ['low','high']]
    axs[1].boxplot(data,labels=['Low','High'],showfliers=False)
    jitter=np.random.default_rng(7).uniform(-.12,.12,100)
    for j,a in enumerate(data): axs[1].scatter(j+1+jitter,a,s=13,alpha=.55,color=COLORS[j+1])
    axs[1].axhline(0,color='#334155',ls='--',lw=1); axs[1].set_ylabel('Per-case token reduction (%)'); axs[1].set_title('100 paired cases; negative = more tokens')
    fig.supxlabel('*Cost follows stored pricing. Latency sums API time, not parallel wall-clock time.',fontsize=9)
    save(fig,'01_efficiency')
    fig,axs=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    variants=['full','low','high']
    for j,v in enumerate(variants):
        s=stats['variants'][v]; axs[0].bar(j,s['final_empty_count'],color=COLORS[j]); axs[0].text(j,s['final_empty_count']+.2,str(s['final_empty_count']),ha='center')
    axs[0].set_xticks(range(3),['Full','Low','High']); axs[0].set_ylim(0,13); axs[0].set_title('Empty final outputs / 100')
    for j,v in enumerate(['low','high']):
        transitions=stats['variants'][v]['transitions']; mat=np.array([[transitions[a+'_to_'+b] for b in ['nonempty','empty']] for a in ['nonempty','empty']])
        ax=axs[j+1]; ax.imshow(mat,cmap='Blues',vmin=0,vmax=100)
        for y in range(2):
            for x in range(2): ax.text(x,y,str(mat[y,x]),ha='center',va='center',color='white' if mat[y,x]>50 else '#152536',fontsize=16)
        ax.set_xticks([0,1],['Nonempty','Empty']); ax.set_yticks([0,1],['Nonempty','Empty']); ax.set_xlabel(v.title()+' final output'); ax.set_ylabel('Full final output'); ax.set_title('Paired output transitions')
    fig.supxlabel('Nonempty is a structural validity check, not verified answer quality.',fontsize=9)
    save(fig,'02_output_validity')
    fn=[n for n in nodes if n['variant']=='full']
    fig,axs=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    names=list(STAGES.values()); ent=[[n['entropy'] for n in fn if n['stage']==s and n['entropy'] is not None] for s in names]
    axs[0].boxplot(ent,labels=names,showfliers=False); axs[0].set_yscale('log'); axs[0].set_ylabel('Normalized top-k entropy (bits/token, log scale)'); axs[0].set_title('Observed entropy by generation stage')
    available=[len(a)/500*100 for a in ent]; axs[1].bar(names,available,color='#4879a5'); axs[1].set_ylim(0,100); axs[1].set_ylabel('Nodes with entropy (%)'); axs[1].set_title('Entropy coverage; 500 slots per stage')
    for i,n in enumerate(available): axs[1].text(i,n+2,f'{n:.1f}%',ha='center')
    for j,v in enumerate(['low','high']):
        s=stats['variants'][v]; counts=s['selected_stage_counts']; axs[2].bar(np.arange(4)+(j-.5)*.36,[counts.get(k,0) for k in names],.36,label=v.title(),color=COLORS[j+1])
    axs[2].set_xticks(range(4),names); axs[2].set_ylabel('Selected skip candidates'); axs[2].set_title('Where each policy selects skips'); axs[2].legend()
    fig.supxlabel('Only original Full generation nodes counted. Missing entropy is not treated as zero.',fontsize=9)
    save(fig,'03_entropy_and_selection')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    full_nonempty=[c for c in cases if c['variant']=='full' and not c['final_empty']]
    for j,v in enumerate(['low','high']):
        pairs=[(f,idx[(f['case_id'],v)]) for f in full_nonempty if not idx[(f['case_id'],v)]['final_empty']]
        axs[0].scatter([a['final_score'] for a,b in pairs],[b['final_score'] for a,b in pairs],s=24,alpha=.55,color=COLORS[j+1],label=f'{v.title()} (n={len(pairs)})')
        axs[1].scatter([100*(1-b['total_tokens']/a['total_tokens']) for a,b in pairs],[b['final_score']-a['final_score'] for a,b in pairs],s=24,alpha=.55,color=COLORS[j+1],label=v.title())
    axs[0].plot([0,10],[0,10],ls='--',color='#64748b'); axs[0].set_xlabel('Full internal score'); axs[0].set_ylabel('Compressed internal score'); axs[0].set_title('Both outputs nonempty'); axs[0].legend()
    axs[1].axhline(0,color='#64748b',ls='--'); axs[1].axvline(0,color='#64748b',ls='--'); axs[1].set_xlabel('Token reduction (%)'); axs[1].set_ylabel('Internal score change vs Full'); axs[1].set_title('Efficiency vs internal score'); axs[1].legend()
    fig.supxlabel('Internal model score is an imperfect proxy; zero may reflect score parsing failure.',fontsize=9)
    save(fig,'04_score_tradeoff_exploratory')
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    ids=stats['variants']['low']['zero_skip_ids']
    for j,v in enumerate(['low','high']):
        values=[100*(1-idx[(i,v)]['total_tokens']/idx[(i,'full')]['total_tokens']) for i in ids]
        ax.bar(np.arange(len(ids))+(j-.5)*.36,values,.36,label=v.title()+' replay',color=COLORS[j+1])
    ax.set_xticks(range(len(ids)),[str(i) for i in ids]); ax.axhline(0,color='#64748b',ls='--')
    ax.set_xlabel('Case ID (zero selected and executed skips)'); ax.set_ylabel('Token reduction vs Full (%)')
    ax.set_title('Changes remain when no nodes are skipped'); ax.legend()
    fig.supxlabel('Selected subgroup, n=10. Shows rerun variability; not a representative randomized control.',fontsize=9)
    save(fig,'05_zero_skip_reruns')


if __name__ == '__main__':
    main()
