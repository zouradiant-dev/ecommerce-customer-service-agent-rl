"""Evaluate evidence recall separately from model answers; dev-only input."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knowledge-file',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--as-of',required=True)
    args=parser.parse_args();index=KnowledgeIndex(args.knowledge_file)
    cases=[json.loads(line) for line in args.cases.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not cases or any(c.get('split')!='dev' for c in cases):raise ValueError('retrieval_runner_accepts_dev_only')
    results=[]
    for case in cases:
        hits=index.search(case['query'],as_of=args.as_of)['hits'];actual={hit['citation'] for hit in hits}
        expected=set(case['expected_citations'])
        results.append({'case_id':case['case_id'],'retrieved_citations':[h['citation'] for h in hits],
            'recall_at_3':len(expected&actual)/len(expected) if expected else None,
            'abstention_correct':not hits if not expected else None})
    positives=[r['recall_at_3'] for r in results if r['recall_at_3'] is not None]
    negatives=[r['abstention_correct'] for r in results if r['abstention_correct'] is not None]
    report={'index_sha256':index.sha256,'cases_sha256':sha256(args.cases.read_bytes()).hexdigest(),'as_of':args.as_of,
        'cases':results,'mean_recall_at_3':sum(positives)/len(positives) if positives else None,
        'no_evidence_abstention_rate':sum(negatives)/len(negatives) if negatives else None,
        'model_answer_evaluated':False,'independent_test_claim':False,'production_quality_claim':False,
        'limitations':['tiny_synthetic_dev_fixture','lexical_retrieval_only_no_embeddings_or_reranker']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:json.dump(report,out,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('mean_recall_at_3','no_evidence_abstention_rate','model_answer_evaluated')}))
