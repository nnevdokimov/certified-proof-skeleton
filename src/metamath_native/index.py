"""Discrimination tree for possible typed one-way subsumers.

Wildcard paths only produce candidates. The caller must still check repeated
variables and every mandatory $d restriction using an explicit substitution.
"""
from .terms import sort


class PatternIndex:
    def __init__(self):self.tree={}

    def add(self,formula,nid):
        node=self.tree
        def visit(t):
            nonlocal node
            key=('*',t[1]) if t[0]=='v' else ('tag',t[0])
            node=node.setdefault(key,{})
            if t[0]!='v':
                for x in t[1:]:visit(x)
        visit(formula);node.setdefault('ids',[]).append(nid)

    def candidates(self,formula):
        terms=[];ends=[]
        def visit(t):
            pos=len(terms);terms.append(t);ends.append(None)
            if t[0]!='v':
                for child in t[1:]:visit(child)
            ends[pos]=len(terms)
        visit(formula);pending=[(self.tree,0)];ids=[]
        while pending:
            node,pos=pending.pop()
            if pos==len(terms):
                ids.extend(node.get('ids',[]));continue
            t=terms[pos];wild=node.get(('*',sort(t)))
            if wild is not None:pending.append((wild,ends[pos]))
            if t[0]!='v':
                rigid=node.get(('tag',t[0]))
                if rigid is not None:pending.append((rigid,pos+1))
        return sorted(ids)
