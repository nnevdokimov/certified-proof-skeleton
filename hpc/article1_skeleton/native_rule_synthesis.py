"""Generalise observed MP proof programmes and discharge their boundary holes.

No human statement enters. Abstract the shape of a short stored MP fragment,
consider every equality partition of its boundary, jointly unify its MP
constraints, and discharge every boundary order by the Hilbert deduction
theorem. Compile the resulting closed proof using the existing ax-1, ax-2,
and ax-mp kernel, with explicit parent substitutions. Gen cuts the fragment:
we do NOT apply unrestricted deduction across a generalisation step.

The finite programme bound is a development resource parameter, not a fixed
dictionary K. This module establishes proof validity, not new mathematical
theorems or computational superiority.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
import time

from metamath_native.index import PatternIndex
from metamath_native.saturation import instance_witness
from metamath_native.terms import variable, variables, syntax, size, unify, canonicalise


def internal_nodes(shape):
    return 0 if shape == 'H' else 1+sum(internal_nodes(s) for s in shape)


def boundary_count(shape):
    return 1 if shape == 'H' else sum(boundary_count(s) for s in shape)


def observed_programmes(lib, maximum_mp=2):
    if not 0 <= maximum_mp <= 3:
        raise ValueError('This finite pilot supports at most three MP nodes')
    counts=Counter({'H':len(lib.nodes)})
    def shape(i,depth):
        n=lib.nodes[i]
        if depth==0 or n.assertion!='ax-mp':return 'H'
        return tuple(shape(p,depth-1) for p,_ in n.parents)
    for n in lib.nodes:
        for depth in range(1,maximum_mp+1):
            s=shape(n.id,depth)
            if 0<internal_nodes(s)<=maximum_mp:counts[s]+=1
    return counts


def boundary_partitions(n):
    """All set partitions, once each, as restricted-growth strings."""
    if n<1:raise ValueError('A programme needs a boundary')
    def visit(prefix):
        if len(prefix)==n:
            yield tuple(prefix);return
        for j in range(max(prefix)+2):yield from visit(prefix+[j])
    yield from visit([0])


@dataclass(eq=False, frozen=True)
class Proof:
    kind: str
    formula: tuple
    holes: frozenset
    children: tuple=()
    data: tuple=()


class ProofFactory:
    def __init__(self):self.cache={};self.deductions={}
    def intern(self,key,create):
        if key not in self.cache:self.cache[key]=create()
        return self.cache[key]
    def hypothesis(self,j,formula):
        return self.intern(('hyp',j,formula),lambda:Proof('hyp',formula,frozenset([j]),data=(j,)))
    def axiom1(self,a,b):
        f=syntax('wi',a,syntax('wi',b,a))
        return self.intern(('ax1',a,b),lambda:Proof('axiom',f,frozenset(),data=('ax-1',)))
    def axiom2(self,a,b,c):
        f=syntax('wi',syntax('wi',a,syntax('wi',b,c)),
            syntax('wi',syntax('wi',a,b),syntax('wi',a,c)))
        return self.intern(('ax2',a,b,c),lambda:Proof('axiom',f,frozenset(),data=('ax-2',)))
    def mp(self,minor,major):
        if major.formula[0]!='wi' or major.formula[1]!=minor.formula:
            raise ValueError('Open proof has a mismatched MP boundary')
        return self.intern(('mp',minor,major),lambda:Proof('mp',major.formula[2],
            minor.holes|major.holes,(minor,major)))
    def identity(self,a):
        return self.mp(self.axiom1(a,a),self.mp(self.axiom1(a,syntax('wi',a,a)),
            self.axiom2(a,syntax('wi',a,a),a)))
    def discharge(self,proof,j,assumption):
        key=proof,j,assumption
        if key in self.deductions:return self.deductions[key]
        if proof.kind=='hyp' and proof.data[0]==j:
            result=self.identity(assumption)
        elif j not in proof.holes:
            result=self.mp(proof,self.axiom1(proof.formula,assumption))
        elif proof.kind=='mp':
            minor,major=proof.children
            left=self.discharge(minor,j,assumption)
            right=self.discharge(major,j,assumption)
            result=self.mp(left,self.mp(right,self.axiom2(assumption,minor.formula,proof.formula)))
        else:raise AssertionError('Unsupported open inference')
        assert result.formula==syntax('wi',assumption,proof.formula)
        assert j not in result.holes
        self.deductions[key]=result
        return result


def abstract_programme(shape,partition,factory):
    if len(partition)!=boundary_count(shape):raise ValueError('Wrong partition length')
    holes={j:variable('wff','programme.h'+str(j)) for j in set(partition)}
    leaves=iter(partition);constraints=[];serial=0
    def prepare(s):
        nonlocal serial
        if s=='H':
            j=next(leaves);return ('hyp',j,holes[j])
        minor,major=(prepare(t) for t in s)
        output=variable('wff','programme.n'+str(serial));serial+=1
        constraints.append((major[-1],syntax('wi',minor[-1],output)))
        return ('mp',minor,major,output)
    raw=prepare(shape)
    def pack(terms):
        out=syntax('wtru')
        for t in reversed(terms):out=syntax('wi',t,out)
        return out
    resolve=unify(pack([a for a,b in constraints]),pack([b for a,b in constraints]))
    def build(t):
        if t[0]=='hyp':return factory.hypothesis(t[1],resolve(t[2]))
        return factory.mp(build(t[1]),build(t[2]))
    return build(raw),{j:resolve(t) for j,t in holes.items()}


class ClosedCompiler:
    """Use existing closed schemas through explicit checked substitutions."""
    def __init__(self,lib,formula_size=160,proof_height=256):
        self.lib=lib;self.formula_size=formula_size;self.proof_height=proof_height
        self.index=PatternIndex();self.memo={};self.counts=Counter()
        self.axioms={n.assertion:n.id for n in lib.nodes if not n.parents}
        for n in lib.nodes:self.index.add(n.formula,n.id)
    def existing(self,formula):
        for i in self.index.candidates(formula):
            self.counts['one_way_checks']+=1;n=self.lib.nodes[i]
            if instance_witness(n.formula,n.disjoint,formula,()) is not None:return i
        return None
    def compile(self,p):
        if p.holes:raise ValueError('Cannot export an undischarged hypothesis')
        if p in self.memo:return self.memo[p]
        found=self.existing(p.formula)
        if found is not None:
            self.counts['reused_closed_schemas']+=1;self.memo[p]=found;return found
        if size(p.formula)>self.formula_size:raise ValueError('Compiled intermediate exceeds size bound')
        if p.kind!='mp':raise AssertionError('An original axiom should already cover its instance')
        minor,major=p.children
        a,b=self.compile(minor),self.compile(major)
        if 1+max(self.lib.nodes[i].height for i in (a,b))>self.proof_height:
            raise ValueError('Compiled proof exceeds height bound')
        parents=[]
        for i,wanted in ((a,minor.formula),(b,major.formula)):
            n=self.lib.nodes[i]
            mapping=instance_witness(n.formula,n.disjoint,wanted,())
            if mapping is None:raise AssertionError('Closed compiler lost a parent instance')
            parents.append((i,{v[2]:mapping[v] for v in variables(n.formula)}))
        origins=sorted(set(self.lib.nodes[a].origins)|set(self.lib.nodes[b].origins))
        self.counts['native_mp_requests']+=1
        nid,fresh=self.lib.add(p.formula,(),'ax-mp',{'ph':minor.formula,'ps':p.formula},parents,origins)
        if fresh:
            self.index.add(self.lib.nodes[nid].formula,nid);self.counts['native_mp_created']+=1
        self.memo[p]=nid
        return nid


def synthesise_rules(lib,maximum_mp=2,formula_size=160,proof_height=256):
    begin=time.process_time();initial=len(lib.nodes)
    programmes=observed_programmes(lib,maximum_mp)
    factory=ProofFactory();compiler=ClosedCompiler(lib,formula_size,proof_height)
    records=[];outputs=set();targets=set();invalid=0
    for shape,count in sorted(programmes.items(),key=lambda x:(internal_nodes(x[0]),repr(x[0]))):
        for partition in boundary_partitions(boundary_count(shape)):
            try:proof,holes=abstract_programme(shape,partition,factory)
            except ValueError:
                invalid+=1;continue
            for order in permutations(sorted(holes)):
                closed=proof
                for j in order:closed=factory.discharge(closed,j,holes[j])
                assert not closed.holes
                target=canonicalise(closed.formula)[0]
                if target in targets:continue
                targets.add(target)
                row=dict(shape=shape,observed_occurrences=count,partition=partition,
                         discharge_order=order,formula=target)
                try:
                    nid=compiler.compile(closed);outputs.add(nid)
                    row.update(status='verified_native',node=nid,previously_archived=nid<initial)
                except ValueError as error:row.update(status='bounded_out',reason=str(error))
                records.append(row)
    for n in lib.nodes:lib.check(n)
    return sorted(outputs),dict(schema='native-proof-programme-synthesis/v1',
        maximum_open_mp=maximum_mp,initial_nodes=initial,final_nodes=len(lib.nodes),
        motifs=[dict(shape=s,occurrences=n) for s,n in programmes.items()],
        invalid_boundary_partitions=invalid,distinct_closed_candidates=len(targets),
        selected_existing_or_new_schemas=sorted(outputs),
        new_rule_schema_count=sum(i>=initial for i in outputs),
        compiler_counters=dict(compiler.counts),records=records,
        cpu_seconds=time.process_time()-begin,
        bounds=dict(formula_size=formula_size,proof_height=proof_height),
        scope='Abstract observed MP programmes and all boundary orders. Deduction uses ax-1/ax-2/ax-mp only; Gen stays outside open fragments. No external theorem targets or new kernel rule. Generality is structural, not human mathematical significance.')
