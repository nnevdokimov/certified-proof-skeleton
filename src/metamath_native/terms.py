"""Typed syntax trees for the primitive ZFC vocabulary used by the chosen basis.

All variables remain Metamath metavariables. Disjoint-variable constraints,
not an AST alpha-renaming convention, govern allowed variable substitutions.
"""
SORTS={'wn':'wff','wi':'wff','wb':'wff','wa':'wff','wo':'wff','wal':'wff',
       'wex':'wff','cv':'class','wceq':'wff','wcel':'wff','wtru':'wff','wfal':'wff'}
ARGS={'wn':('wff',),'wi':('wff','wff'),'wb':('wff','wff'),'wa':('wff','wff'),
      'wo':('wff','wff'),'wal':('setvar','wff'),'wex':('setvar','wff'),
      'cv':('setvar',),'wceq':('class','class'),'wcel':('class','class'),'wtru':(),'wfal':()}
BINARY={'->':'wi','<->':'wb','/\\':'wa','\\/':'wo'}
REVERSE={v:k for k,v in BINARY.items()}


def validate(t):
    import re
    if not isinstance(t,tuple) or not t:raise ValueError('Malformed typed term')
    if t[0]=='v':
        if (len(t)!=3 or t[1] not in ('wff','class','setvar') or
                not isinstance(t[2],str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*',t[2])):
            raise ValueError('Malformed typed variable')
        return t[1]
    if t[0] not in ARGS or len(t)-1!=len(ARGS[t[0]]):raise ValueError('Unknown constructor or arity')
    if tuple(validate(x) for x in t[1:])!=ARGS[t[0]]:raise ValueError('Ill-typed child')
    return SORTS[t[0]]


def variable(sort,name):return ('v',sort,name)
def sort(t):return t[1] if t[0]=='v' else SORTS[t[0]]
def variables(t):
    if t[0]=='v':return {t}
    return set().union(*(variables(x) for x in t[1:])) if len(t)>1 else set()
def size(t):return 1 if t[0]=='v' else 1+sum(size(x) for x in t[1:])


def syntax(tag,*children):
    if tag not in ARGS or tuple(map(sort,children))!=ARGS[tag]:raise ValueError('Ill-typed constructor '+tag)
    return (tag,*children)


def parse(tokens,types):
    tokens=tuple(tokens);i=0
    def cls():
        nonlocal i
        name=tokens[i];i+=1
        if types.get(name)=='class':return variable('class',name)
        if types.get(name)=='setvar':return syntax('cv',variable('setvar',name))
        raise ValueError('Unsupported class expression '+name)
    def wff():
        nonlocal i
        token=tokens[i];i+=1
        if types.get(token)=='wff':return variable('wff',token)
        if token=='-.':return syntax('wn',wff())
        if token in ('A.','E.'):
            name=tokens[i];i+=1
            if types.get(name)!='setvar':raise ValueError('Quantifier variable must have setvar type')
            return syntax('wal' if token=='A.' else 'wex',variable('setvar',name),wff())
        if token in ('T.','F.'):return syntax('wtru' if token=='T.' else 'wfal')
        if token=='(':
            left=wff();op=tokens[i];i+=1;right=wff()
            if tokens[i]!=')':raise ValueError('Missing closing parenthesis')
            i+=1;return syntax(BINARY[op],left,right)
        i-=1;left=cls();op=tokens[i];i+=1;right=cls()
        if op not in ('=','e.'):raise ValueError('Unsupported atomic relation')
        return syntax('wceq' if op=='=' else 'wcel',left,right)
    result=wff()
    if i!=len(tokens):raise ValueError('Trailing tokens')
    return result


def render(t):
    tag=t[0]
    if tag=='v':return (t[2],)
    if tag=='cv':return render(t[1])
    if tag=='wn':return ('-.',)+render(t[1])
    if tag in REVERSE:return ('(',)+render(t[1])+(REVERSE[tag],)+render(t[2])+(')',)
    if tag in ('wal','wex'):return ('A.' if tag=='wal' else 'E.',)+render(t[1])+render(t[2])
    if tag in ('wceq','wcel'):return render(t[1])+('=' if tag=='wceq' else 'e.',)+render(t[2])
    if tag in ('wtru','wfal'):return ('T.' if tag=='wtru' else 'F.',)
    raise ValueError(tag)


def replace(t,mapping):
    if t[0]=='v':return mapping.get(t,t)
    return (t[0],*(replace(x,mapping) for x in t[1:]))


def renamed(t,prefix):return replace(t,{v:variable(v[1],prefix+str(v[2])) for v in variables(t)})


def unify(a,b):
    """Typed first-order MGU with occurs check; does not discard $d conditions."""
    subst={};pending=[(a,b)]
    def resolve(x):
        if x[0]=='v':return resolve(subst[x]) if x in subst else x
        return (x[0],*(resolve(y) for y in x[1:]))
    while pending:
        x,y=map(resolve,pending.pop())
        if x==y:continue
        if sort(x)!=sort(y):raise ValueError('Type mismatch')
        if y[0]=='v' and x[0]!='v':x,y=y,x
        if x[0]=='v':
            if x in variables(y):raise ValueError('Occurs check')
            subst[x]=y
        elif x[0]!=y[0]:raise ValueError('Constructor mismatch')
        else:pending.extend(zip(x[1:],y[1:]))
    return resolve


def canonicalise(conclusion, proof_expressions=()):
    mapping={};counts={'wff':0,'setvar':0,'class':0};prefix={'wff':'np','setvar':'nx','class':'nC'}
    def visit(t):
        if t[0]=='v':
            if t not in mapping:
                k=counts[t[1]];counts[t[1]]+=1;mapping[t]=variable(t[1],prefix[t[1]]+str(k))
            return
        for child in t[1:]:visit(child)
    visit(conclusion)
    for t in proof_expressions:visit(t)
    return replace(conclusion,mapping),mapping


def mapped_disjoint(disjoint, substitution):
    result=set()
    for a,b in disjoint:
        left=variables(substitution[a]);right=variables(substitution[b])
        if left & right:raise ValueError('Disjoint variable violation')
        result.update(tuple(sorted((x,y))) for x in left for y in right)
    return result
