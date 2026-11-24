"""Check committed proofs, canonicalize first proofs, and run the common selector."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from certificate import axioms,mp
from fragment_policy import ProofCoverage
from functools import lru_cache

def encoded(t):
    return str(t[1])+'.' if t[0]=='v' else t[0]+''.join(encoded(c) for c in t[1:])

class FirstProofs:
    def __init__(self):
        ax=axioms('prop')
        self.forms=[ax['ax-'+str(i)] for i in (1,2,3)]
        self.by_formula={f:i for i,f in enumerate(self.forms)}
        self.parents=[(),(),()]
        self.retained=set()
    def detach(self,major,minor):
        f=mp(self.forms[minor],self.forms[major])
        node=self.by_formula.get(f)
        if node is None:
            node=len(self.forms)
            self.by_formula[f]=node
            self.forms.append(f)
            self.parents.append((major,minor))
        return node

def read_archive(path,arm):
    g=FirstProofs();native=[];records=0;partial=0
    @lru_cache(maxsize=None)
    def dterm(text):
        if text in ('1','2','3'):return int(text)-1
        if not (text.startswith('d(') and text.endswith(')')):raise ValueError(text)
        level=0
        for k in range(2,len(text)-1):
            c=text[k]
            if c==',' and level==0:
                return g.detach(dterm(text[2:k]),dterm(text[k+1:-1]))
            level+=int(c=='(')-int(c==')')
        raise ValueError(text)
    with gzip.open(path,'rt') as f:
        for line in f:
            if not line.endswith('\n'):
                partial+=1;continue
            fields=line.rstrip('\n').split('\t')
            if fields[0]!='K':raise ValueError(line[:100])
            if arm=='ours':
                _,seq,proof,formula=fields; assert int(seq)==len(native)
                if proof.startswith('ax-'):
                    node=int(proof[3:])-1;assert 0<=node<3
                else:
                    a,b=map(int,proof.split(','));node=g.detach(native[a],native[b])
                native.append(node)
            else:
                _,proof,formula=fields;node=dterm(proof)
            assert encoded(g.forms[node])==formula,(path,records,formula)
            if node>=3:g.retained.add(node)
            records+=1
    assert g.retained
    return g,dict(complete_records=records,partial_final_records=partial,
                  retained_derived=len(g.retained),internal_only=len(g.forms)-3-len(g.retained),
                  distinct_certified_formulas=len(g.forms),raw_distinct_terms=(dterm.cache_info().currsize if arm=='sgcd' else len(native)))

def select(g):
    records={i:(i,g.forms[i],None,p) for i,p in enumerate(g.parents)}
    policy=ProofCoverage()
    policy.update(records)
    _,report=policy.freeze(records,visible_proofs={i-3 for i in g.retained})
    ids=[row['root'] for row in report['trace']]
    support=set();todo=list(ids)
    while todo:
        i=todo.pop()
        if i not in support:support.add(i);todo.extend(g.parents[i])
    return dict(retained_training_proofs=len(g.retained),covered_training_proofs=report['covered_proofs'],
        eligible_fragments=report['candidates_count'],frontier_claims=report['frontier_claims'],anchors=[dict(id=i,formula=encoded(g.forms[i]),
        height=policy.heights[i],marginal_roots=row['additional_proofs'],frontier_consumers=row['frontier_consumers'],marginal_votes=row['marginal_votes']) for i,row in zip(ids,report['trace'])],
        definition_steps=sum(i>=3 for i in support),proof_dag={str(i):list(g.parents[i]) for i in sorted(support)})

def main(trace,dest,name):
    arm='ours' if name.startswith('ours') else 'sgcd'
    graph,meta=read_archive(trace,arm)
    bank=dict(run=name,**meta,**select(graph),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    with dest.open('x') as f:json.dump(bank,f,indent=2);f.write('\n')
    print(name,meta['retained_derived'],len(bank['anchors']),bank['definition_steps'],flush=True)
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
