"""Extract an explicit axiom/definition basis, preserving native Metamath frames."""
from dataclasses import dataclass
import hashlib
from itertools import combinations
from pathlib import Path
import re

LOGIC = ['ax-mp'] + [f'ax-{i}' for i in range(1, 14)] + ['ax-gen']
ZFC = ['ax-ext','ax-rep','ax-sep','ax-nul','ax-pow','ax-pr','ax-un','ax-reg','ax-inf','ax-ac']
DEFINITIONS = ['df-bi','df-an','df-or','df-ex','df-tru','df-fal']
SYNTAX = ['wn','wi','wb','wa','wo','wal','wex','cv','wceq','wcel','wtru','wfal']
LABELS = set(LOGIC + ZFC + DEFINITIONS + SYNTAX)


@dataclass(frozen=True)
class Assertion:
    label: str
    kind: str
    expression: tuple
    floats: tuple
    essentials: tuple
    disjoint: tuple

    @property
    def types(self):
        return {expression[1]: expression[0] for _,expression in self.floats}


def load_basis(path):
    raw=Path(path).read_text()
    tokens=re.sub(r'\$\(.*?\$\)', ' ', raw, flags=re.S).split()
    if '$(' in tokens or '$)' in tokens:raise ValueError('Unclosed or stray comment')
    active_variables=set();hypotheses=[];dv=set();scopes=[];selected=[]
    i=0;label=None
    while i<len(tokens):
        t=tokens[i]
        if t=='${':
            scopes.append((set(active_variables),len(hypotheses),set(dv)));label=None;i+=1;continue
        if t=='$}':
            active_variables,n,dv=scopes.pop();hypotheses=hypotheses[:n];label=None;i+=1;continue
        if t in ('$c','$v','$d','$f','$e','$a','$p'):
            end=tokens.index('$.',i+1);body=tokens[i+1:end]
            if t=='$v':active_variables.update(body)
            elif t=='$d':dv.update(tuple(sorted(x)) for x in combinations(body,2))
            elif t in ('$f','$e'):hypotheses.append((t,label,tuple(body)))
            elif t=='$a' and label in LABELS:
                needed=active_variables.intersection(body)
                essentials=tuple((name,e) for k,name,e in hypotheses if k=='$e')
                for _,e in essentials:needed.update(active_variables.intersection(e))
                floats=tuple((name,e) for k,name,e in hypotheses if k=='$f' and e[1] in needed)
                if {e[1] for _,e in floats}!=needed:raise ValueError('Missing floating hypothesis')
                selected.append(Assertion(label,'syntax' if body[0]!='|-' else 'logical',tuple(body),
                    floats,essentials,tuple(sorted(x for x in dv if set(x)<=needed))))
            elif t=='$p' and label in LABELS:raise ValueError('Requested axiom is actually a theorem: '+label)
            label=None;i=end+1;continue
        if t=='$[':raise ValueError('External includes unsupported; use the pinned standalone set.mm')
        label=t;i+=1
    if scopes:raise ValueError('Unclosed scope')
    if {a.label for a in selected}!=LABELS:raise ValueError('Missing basis labels: '+str(LABELS-{a.label for a in selected}))
    return selected,hashlib.sha256(raw.encode()).hexdigest()


def basis_text(assertions, extra_types=None):
    """Fresh standalone file; scoped $d restrictions never become global."""
    declarations={}
    for a in assertions:
        for label,expr in a.floats:
            if expr[1] in declarations and declarations[expr[1]][1]!=expr[0]:
                raise ValueError('Variable changes floating type')
            # set.mm redeclares a few local $f labels (vx.wal, vx.cv, ...).
            # Use one global typed declaration; no imported $p refers to them.
            declarations.setdefault(expr[1],(label,expr[0]))
    for variable,sort in (extra_types or {}).items():
        if variable in declarations:raise ValueError('Extra variable collides with basis')
        declarations[variable]=('native.f.'+variable,sort)
    constants={'|-','wff','setvar','class'}
    for a in assertions:
        constants.update(x for x in a.expression if x not in a.types)
        for _,e in a.essentials:constants.update(x for x in e if x not in a.types)
    lines=['$( Native basis extracted from pinned set.mm; no imported $p statements. $)',
        '$c '+' '.join(sorted(constants))+' $.', '$v '+' '.join(declarations)+' $.']
    for variable,(label,sort) in declarations.items():lines.append(f'{label} $f {sort} {variable} $.')
    for a in assertions:
        lines.append('${')
        for x,y in a.disjoint:lines.append(f'$d {x} {y} $.')
        for label,e in a.essentials:lines.append(label+' $e '+' '.join(e)+' $.')
        lines.append(a.label+' $a '+' '.join(a.expression)+' $.')
        lines.append('$}')
    return '\n'.join(lines)+'\n',declarations
