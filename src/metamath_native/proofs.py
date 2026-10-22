"""Typed certificates with explicit substitutions and native $d obligations.

Metamath's standalone verifier remains the independent acceptance check for
exported libraries. This module does not import the historical proof generator.
"""
from collections import Counter
from dataclasses import asdict,dataclass

from .terms import (variables,variable,sort,syntax,parse,render,replace,renamed,
                    unify,canonicalise,mapped_disjoint,validate)


def freeze(x):return tuple(freeze(y) for y in x) if isinstance(x,(list,tuple)) else x
def ordered_vars(t):return sorted(variables(t),key=lambda v:(v[1],v[2]))


@dataclass
class Node:
    id: int
    formula: tuple
    disjoint: tuple
    proof_disjoint: tuple
    assertion: str
    arguments: dict
    parents: tuple
    height: int
    origins: tuple

    @property
    def label(self):return 'native.'+str(self.id)
    @property
    def key(self):return self.formula,self.disjoint


class Library:
    def __init__(self,assertions):
        self.basis=assertions;self.by_label={a.label:a for a in assertions}
        self.nodes=[];self.known={};self.reuse=Counter()

    def add(self,formula,dv,assertion,arguments,parents=(),origins=()):
        proof_terms=list(arguments.values())
        for _,mapping in parents:proof_terms.extend(mapping.values())
        conclusion,mapping=canonicalise(formula,proof_terms)
        # Every $d variable must occur in a supplied proof expression.
        if any(x not in mapping for pair in dv for x in pair):raise ValueError('Unaccounted proof variable')
        all_dv=tuple(sorted({tuple(sorted((mapping[a],mapping[b]))) for a,b in dv}))
        mandatory=variables(conclusion)
        frame_dv=tuple(pair for pair in all_dv if set(pair)<=mandatory)
        key=conclusion,frame_dv
        if key in self.known:return self.known[key],False
        args={name:replace(value,mapping) for name,value in arguments.items()}
        uses=tuple((i,{name:replace(value,mapping) for name,value in args.items()}) for i,args in parents)
        nid=len(self.nodes)
        node=Node(nid,conclusion,frame_dv,all_dv,assertion,args,uses,
            1+max((self.nodes[i].height for i,_ in uses),default=-1),tuple(sorted(set(origins))))
        self.check(node)
        self.nodes.append(node);self.known[key]=nid;self.reuse.update({i for i,_ in uses})
        return nid,True

    def initialise(self):
        for a in self.basis:
            if a.kind!='logical' or a.essentials:continue
            f=parse(a.expression[1:],a.types)
            args={name:variable(kind,name) for name,kind in a.types.items()}
            dv={(args[x],args[y]) for x,y in a.disjoint}
            self.add(f,dv,a.label,args,origins=[a.label])

    def check(self,node):
        if validate(node.formula)!='wff':raise ValueError('Claim must have wff syntax')
        for t in node.arguments.values():validate(t)
        for _,args in node.parents:
            for t in args.values():validate(t)
        for pair in node.proof_disjoint:
            if len(pair)!=2:raise ValueError('Malformed $d pair')
            for t in pair:
                if t[0]!='v':raise ValueError('$d requires variables')
                validate(t)
        a=self.by_label[node.assertion]
        if a.kind!='logical' or len(a.essentials)!=len(node.parents):raise ValueError('Incorrect assertion frame')
        args=node.arguments
        if set(args)!=set(a.types):raise ValueError('Missing or extra axiom arguments')
        if any(sort(args[name])!=kind for name,kind in a.types.items()):raise ValueError('Argument type mismatch')
        original={variable(kind,name):args[name] for name,kind in a.types.items()}
        claimed=replace(parse(a.expression[1:],a.types),original)
        if claimed!=node.formula:raise ValueError('Wrong claimed conclusion')
        obligations=mapped_disjoint([(variable(a.types[x],x),variable(a.types[y],y)) for x,y in a.disjoint],original)
        for (_,essential),(parent_id,mapping) in zip(a.essentials,node.parents):
            if not 0<=parent_id<node.id:raise ValueError('Non-earlier parent')
            parent=self.nodes[parent_id];needed={v[2] for v in variables(parent.formula)}
            if set(mapping)!=needed:raise ValueError('Parent floating arguments incomplete')
            psubst={v:mapping[v[2]] for v in variables(parent.formula)}
            if any(sort(v)!=sort(psubst[v]) for v in psubst):raise ValueError('Parent type mismatch')
            if replace(parent.formula,psubst)!=replace(parse(essential[1:],a.types),original):
                raise ValueError('Essential hypothesis does not match parent')
            obligations.update(mapped_disjoint(parent.disjoint,psubst))
        if set(node.proof_disjoint)!=obligations:raise ValueError('Incorrect or hidden $d obligations')
        mandatory=variables(node.formula)
        expected=tuple(sorted(pair for pair in obligations if set(pair)<=mandatory))
        if node.disjoint!=expected:raise ValueError('Incorrect mandatory $d frame')
        if any(x==y for x,y in node.proof_disjoint):raise ValueError('Reflexive $d pair')
        expected_height=1+max((self.nodes[i].height for i,_ in node.parents),default=-1)
        if node.height!=expected_height:raise ValueError('Incorrect proof height')
        expected_origins=set().union(*(set(self.nodes[i].origins) for i,_ in node.parents)) if node.parents else {a.label}
        if set(node.origins)!=expected_origins:raise ValueError('Incorrect axiom provenance')

    def propose_mp(self,minor_id,major_id):
        minor=self.nodes[minor_id];major=self.nodes[major_id]
        if major.formula[0]!='wi':raise ValueError('Not an implication')
        left=renamed(minor.formula,'left.');right=renamed(major.formula,'right.')
        resolve=unify(left,right[1]);conclusion=resolve(right[2])
        pmap={v:resolve(variable(v[1],'left.'+v[2])) for v in variables(minor.formula)}
        qmap={v:resolve(variable(v[1],'right.'+v[2])) for v in variables(major.formula)}
        dv=mapped_disjoint(minor.disjoint,pmap)|mapped_disjoint(major.disjoint,qmap)
        arguments={'ph':resolve(left),'ps':conclusion}
        uses=[(minor_id,{v[2]:x for v,x in pmap.items()}),(major_id,{v[2]:x for v,x in qmap.items()})]
        return conclusion,dv,'ax-mp',arguments,uses,sorted(set(minor.origins)|set(major.origins))

    def propose_gen(self,parent_id,existing=False):
        parent=self.nodes[parent_id]
        choices=sorted((v for v in variables(parent.formula) if v[1]=='setvar'))
        x=choices[0] if existing and choices else variable('setvar','fresh.generalisation')
        args={'ph':parent.formula,'x':x}
        identity={v:v for v in variables(parent.formula)}
        uses=[(parent_id,{v[2]:v for v in variables(parent.formula)})]
        return syntax('wal',x,parent.formula),mapped_disjoint(parent.disjoint,identity),'ax-gen',args,uses,parent.origins

    def accept(self,proposal):return self.add(*proposal)

    def json_nodes(self):return [asdict(n) for n in self.nodes]

    def project(self,roots):
        """Physically retain only the core's proof closure plus the original basis."""
        keep=set(roots);pending=list(roots)
        while pending:
            for i,_ in self.nodes[pending.pop()].parents:
                if i not in keep:keep.add(i);pending.append(i)
        result=Library(self.basis);result.initialise();mapping={}
        for i in sorted(keep):
            n=self.nodes[i]
            parents=[(mapping[j],args) for j,args in n.parents]
            new,_=result.add(n.formula,n.proof_disjoint,n.assertion,n.arguments,parents,n.origins)
            mapping[i]=new
        return result,[mapping[i] for i in roots],mapping

    @classmethod
    def restore(cls,assertions,rows):
        lib=cls(assertions)
        for row in rows:
            node=Node(row['id'],freeze(row['formula']),freeze(row['disjoint']),freeze(row['proof_disjoint']),
                row['assertion'],{k:freeze(v) for k,v in row['arguments'].items()},
                tuple((i,{k:freeze(v) for k,v in args.items()}) for i,args in row['parents']),
                row['height'],freeze(row['origins']))
            if node.id!=len(lib.nodes) or node.key in lib.known:raise ValueError('Duplicate or non-contiguous node')
            lib.check(node);lib.nodes.append(node);lib.known[node.key]=node.id
            lib.reuse.update({i for i,_ in node.parents})
        return lib


SYNTAX_FORMALS={'wn':('ph',),'wi':('ph','ps'),'wb':('ph','ps'),'wa':('ph','ps'),
    'wo':('ph','ps'),'wal':('x','ph'),'wex':('x','ph'),'cv':('x',),
    'wceq':('A','B'),'wcel':('A','B'),'wtru':(),'wfal':()}


def proof_text(library,selected=None,original_include=None):
    from .basis import basis_text
    ids=set(range(len(library.nodes))) if selected is None else set(selected)
    pending=list(ids)
    while pending:
        for i,_ in library.nodes[pending.pop()].parents:
            if i not in ids:ids.add(i);pending.append(i)
    extra_types={}
    for i in sorted(ids):
        n=library.nodes[i];terms=list(n.arguments.values())
        for _,args in n.parents:terms.extend(args.values())
        for t in terms:
            for v in variables(t):extra_types[v[2]]=v[1]
    extra_types=dict(sorted(extra_types.items(),key=lambda x:(x[1],x[0])))
    base,declarations=basis_text(library.basis,extra_types)
    if original_include is not None:
        if any(c in original_include for c in '\n\r\t '):raise ValueError('Unsafe include path')
        base='$[ '+original_include+' $]\n$v '+' '.join(extra_types)+' $.\n'
        base+='\n'.join(declarations[name][0]+' $f '+kind+' '+name+' $.' for name,kind in extra_types.items())+'\n'
    def syntax_proof(t):
        if t[0]=='v':return [declarations[t[2]][0]]
        a=library.by_label[t[0]];mapping=dict(zip(SYNTAX_FORMALS[t[0]],t[1:]))
        if set(mapping)!=set(a.types):raise ValueError('Unexpected syntax axiom frame')
        result=[]
        for _,(_,name) in a.floats:result.extend(syntax_proof(mapping[name]))
        # Check constructor interpretation against the literal source tokens.
        substituted=tuple(y for token in a.expression[1:] for y in (render(mapping[token]) if token in mapping else (token,)))
        if substituted!=render(t):raise ValueError('Syntax compiler and source axiom disagree')
        return result+[a.label]
    lines=[base]
    for i in sorted(ids):
        node=library.nodes[i];proof=[];a=library.by_label[node.assertion]
        for _,(_,name) in a.floats:proof.extend(syntax_proof(node.arguments[name]))
        for parent_id,args in node.parents:
            for v in ordered_vars(library.nodes[parent_id].formula):proof.extend(syntax_proof(args[v[2]]))
            proof.append(library.nodes[parent_id].label)
        proof.append(a.label)
        lines.append('${')
        for x,y in node.proof_disjoint:lines.append(f'$d {x[2]} {y[2]} $.')
        lines.append(node.label+' $p |- '+' '.join(render(node.formula))+' $= '+' '.join(proof)+' $.')
        lines.append('$}')
    return '\n'.join(lines)+'\n',sorted(ids)
