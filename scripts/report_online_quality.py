"""Build a review queue from dev case traces; no automatic production quality claim."""
import argparse
from collections import Counter
import json
from pathlib import Path


def report(run):
    cases=[]
    for directory in sorted((run/'cases').iterdir()):
        result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        cases.append(result)
    failures=Counter(f for case in cases for f in case['failures'])
    queue=[{'case_id':case['case_id'],'failures':case['failures'],'reviewed':False,
            'evidence_files':[str((run/'cases'/case['case_id']/name).resolve())
                              for name in ('result.json','history.json','events.json','spans.json')]}
           for case in cases if not case['passed']]
    return {'source':'existing_development_run','evaluated_cases':len(cases),
            'dev_passed':sum(case['passed'] for case in cases),'failure_counts':dict(failures),
            'review_queue':queue,'human_reviews_completed':0,'online_production_success_rate':None,
            'note':'开发状态断言生成待复核队列；trace 错误率不等于任务失败率。'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=report(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
    print(json.dumps({'cases':result['evaluated_cases'],'review_queue':len(result['review_queue'])}))
