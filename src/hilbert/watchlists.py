# -*- coding: utf-8 -*-
"""Реестр watchlist'ов (вех) для directed-роста — канонический источник.

Используется run_experiment.py (env ``ZFC_WATCHLIST=<имя>``) и скриптами
фабрики. Имена вех — set.mm; формулы в нашем AST.

Реестр (``get_watchlist``):

- ``setmm_prop_classical`` — 14 пропозициональных теорем set.mm (полный);
- ``setmm_prop_intuit``    — валидные в IPC (без LEM/DNE);
- ``setmm_prop_minimal``   — валидные в минимальной (и без ex falso);
- ``modal_k_basic``        — простые K-вехи (Nec/K-цепочки);
- ``zfc_cantor_native``    — путь к Кантору, native-связки (metamath_zfc):
                             Рассел (диагональ canth) + симметрия равенства;
- ``zfc_cantor_prim``      — то же в примитивной форме (h16_zfc).

Политика: watchlist — часть конфигурации T (фиксируется в metadata
growth_policy). Изменение состава = новое имя или версия.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from core.formula import (
    Box, Equality, Exists, ForAll, Formula, Iff, Implies, Membership, Not,
    PropVar, Term,
)
from core.primitives import to_primitive_form

_p, _q, _r = PropVar("ph"), PropVar("ps"), PropVar("ch")
_x, _y, _z = Term("x"), Term("y"), Term("z")
N, I = Not, Implies

SETMM_PROP_CLASSICAL: List[Tuple[str, Formula]] = [
    ("id", I(_p, _p)),
    ("pm2.43", I(I(_p, I(_p, _q)), I(_p, _q))),
    ("pm2.04", I(I(_p, I(_q, _r)), I(_q, I(_p, _r)))),
    ("imim2", I(I(_p, _q), I(I(_r, _p), I(_r, _q)))),
    ("syl-closed", I(I(_p, _q), I(I(_q, _r), I(_p, _r)))),
    ("pm2.27", I(_p, I(I(_p, _q), _q))),
    ("pm2.21", I(N(_p), I(_p, _q))),
    ("idd", I(_p, I(_q, _q))),
    ("notnot1", I(_p, N(N(_p)))),
    ("notnot2", I(N(N(_p)), _p)),
    ("pm2.18", I(I(N(_p), _p), _p)),
    ("peirce", I(I(I(_p, _q), _p), _p)),
    ("pm2.01", I(I(_p, N(_p)), N(_p))),
    ("con3", I(I(_p, _q), I(N(_q), N(_p)))),
]

_MINIMAL_NAMES = {"id", "pm2.43", "pm2.04", "imim2", "syl-closed",
                  "pm2.27", "idd", "notnot1", "pm2.01", "con3"}
SETMM_PROP_MINIMAL = [m for m in SETMM_PROP_CLASSICAL if m[0] in _MINIMAL_NAMES]
SETMM_PROP_INTUIT = SETMM_PROP_MINIMAL + [
    m for m in SETMM_PROP_CLASSICAL if m[0] == "pm2.21"
]

SETMM_FOL_PREDICATE: List[Tuple[str, Formula]] = [
    ("sp", Implies(ForAll("x", _p), _p)),
    ("19.8a", Implies(_p, Exists("x", _p))),
    ("alcom", Implies(ForAll("x", ForAll("y", _p)),
                      ForAll("y", ForAll("x", _p)))),
    ("alim", Implies(ForAll("x", Implies(_p, _q)),
                     Implies(ForAll("x", _p), ForAll("x", _q)))),
    ("equid", Equality(_x, _x)),
    ("equcomi", Implies(Equality(_x, _y), Equality(_y, _x))),
    ("equtr", Implies(Equality(_x, _y),
                      Implies(Equality(_y, _z), Equality(_x, _z)))),
    ("elequ1", Implies(Equality(_x, _y),
                       Implies(Membership(_x, _z), Membership(_y, _z)))),
]

MODAL_K_BASIC: List[Tuple[str, Formula]] = [
    ("nec-id", Box(I(_p, _p))),
    ("nec-ax1", Box(I(_p, I(_q, _p)))),
    ("nec-nec-id", Box(Box(I(_p, _p)))),
    ("box-mp-id", I(Box(_p), Box(_p))),
]

# ── Глубокая пропозиц. цель (демонстрация направленного роста) ──────────────
# Цепь силлогизма длины 3: (p→q)→((q→r)→((r→s)→(p→s))). Minimal-valid (чистая
# импликация, BCK), но ГЛУБОКАЯ: beam@b64 достигает её на d12 (~67K узлов),
# тогда как слепой BFS до d12 недостижим (взрыв уже на d3–4). _s — 4-й пропвар.
_s, _t, _u = PropVar("th"), PropVar("ta"), PropVar("et")
DEEP_PROP: List[Tuple[str, Formula]] = [
    # Цепи силлогизма растущей длины — minimal-valid, но всё глубже.
    # syl3 достигается beam@b64 на d12; syl4/syl5 — фронт для СК (d16–20).
    ("syl3-chain", I(I(_p, _q), I(I(_q, _r), I(I(_r, _s), I(_p, _s))))),
    ("syl4-chain", I(I(_p, _q), I(I(_q, _r), I(I(_r, _s), I(I(_s, _t), I(_p, _t)))))),
    ("syl5-chain", I(I(_p, _q), I(I(_q, _r), I(I(_r, _s),
                    I(I(_s, _t), I(I(_t, _u), I(_p, _u))))))),
]

# ── Путь к Кантору ────────────────────────────────────────────────────────
# Рассел ¬∃y∀z(z∈y ↔ ¬z∈z) — диагональное ядро canth (следствие
# Sep-инстанса ¬(z∈z)); eqsym — слой равенства (C16); free-вариант — ближняя
# ступень к квантифицированному.
_russell = Not(Exists("y", ForAll("z",
    Iff(Membership(_z, _y), Not(Membership(_z, _z))))))
_eq_sym = ForAll("x", ForAll("y",
    Implies(Equality(_x, _y), Equality(_y, _x))))
_eq_sym_free = Implies(Equality(_x, _y), Equality(_y, _x))

ZFC_CANTOR_NATIVE: List[Tuple[str, Formula]] = [
    ("russell", _russell),
    ("eqsym", _eq_sym),
    ("eqsym-free", _eq_sym_free),
]
ZFC_CANTOR_PRIM = [(n, to_primitive_form(f)) for n, f in ZFC_CANTOR_NATIVE]

_REGISTRY: Dict[str, List[Tuple[str, Formula]]] = {
    "setmm_prop_classical": SETMM_PROP_CLASSICAL,
    "setmm_prop_intuit": SETMM_PROP_INTUIT,
    "setmm_prop_minimal": SETMM_PROP_MINIMAL,
    "setmm_fol_predicate": SETMM_FOL_PREDICATE,
    "modal_k_basic": MODAL_K_BASIC,
    "deep_prop": DEEP_PROP,
    "zfc_cantor_native": ZFC_CANTOR_NATIVE,
    "zfc_cantor_prim": ZFC_CANTOR_PRIM,
}


def get_watchlist(name: str) -> List[Tuple[str, Formula]]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown watchlist {name!r}; available: {sorted(_REGISTRY)}"
        )
    return list(_REGISTRY[name])


def available() -> List[str]:
    return sorted(_REGISTRY)
