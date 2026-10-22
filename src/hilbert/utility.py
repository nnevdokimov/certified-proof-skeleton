# -*- coding: utf-8 -*-
"""Полезность формулы U(φ) для направленного роста графа (directed growth).

Мотивация и литература: docs/DIRECTED_GROWTH_LITERATURE.md (§2–3),
docs/DIRECTING_GROWTH.md (декомпозиция: топология требует отбора к целям).

U(φ) — взвешенная сумма четырёх сигналов, вычислимых на лету без ML:

1. ``resonance``  — резонанс по Wos: совпадение φ с вехой watchlist'а
   точно (canonical hash) или по форме (shape-скелет: все переменные/
   метапеременные стёрты, связки и константы сохранены).
2. ``relevance``  — SInE-стиль символьная релевантность к ближайшей ещё
   НЕ выведенной вехе: взвешенный Jaccard по сигнатурным символам
   (конструкторы, константы, имена предикатов); вес символа обратен его
   частоте в корпусе вех (редкий символ — сильный сигнал).
3. ``rediscovery`` — log(1+r), где r — число независимых переоткрытий φ
   (rediscovery-гиперрёбра): «естественная сходимость» вывода к φ,
   прокси переиспользования (given-clause-by-usage, AUC 0.84 reuse-компаса).
4. ``size``        — штраф −log(size): против веера больших формул.

Политика: ``utility_v1``. Любое изменение состава/весов — инкремент версии.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.formula import Formula, Implies, PropVar, Term
from core.schema import canonicalize_with_hash, match as schema_match

UTILITY_POLICY = "utility_v1"


class AdaptiveBeam:
    """Контроллер ширины бима по стагнации вех (политика adaptive_beam_v1).

    Идея iterative widening: пока вехи падают — узкий глубокий ствол
    (экономия); если ``patience`` раундов нет ни одного нового попадания —
    ширина умножается на ``factor`` (вплоть до ``max_width``); при попадании
    ширина сбрасывается к ``min_width``. Аналог age/weight + restart-стратегий
    сатурационных пруверов.
    """

    def __init__(self, min_width: int = 64, max_width: int = 1024,
                 patience: int = 3, factor: int = 2):
        if min_width < 1 or max_width < min_width:
            raise ValueError("require 1 <= min_width <= max_width")
        self.min_width = min_width
        self.max_width = max_width
        self.patience = patience
        self.factor = factor
        self.width = min_width
        self._stagnant = 0
        self._last_hits = 0

    def next_width(self, hits_total: int) -> int:
        """Вызывается перед активацией раунда; возвращает текущую ширину."""
        if hits_total > self._last_hits:
            self.width = self.min_width
            self._stagnant = 0
        else:
            self._stagnant += 1
            if self._stagnant >= self.patience:
                self.width = min(self.max_width, self.width * self.factor)
                self._stagnant = 0
        self._last_hits = hits_total
        return self.width

    def metadata(self) -> Dict[str, object]:
        return {
            "beam_policy": "adaptive_beam_v1",
            "min_width": self.min_width, "max_width": self.max_width,
            "patience": self.patience, "factor": self.factor,
        }


# ── Сигнатурные символы и shape-скелет ────────────────────────────────────

def _walk_parts(obj, symbols: Set[str], shape: List[str]) -> None:
    """Рекурсивный обход AST: собирает сигнатурные символы и shape-токены.

    Переменные (FO, PropVar, предикатные) стираются в анонимные токены —
    shape совпадает у формул одинаковой формы (резонатор Wos'а).
    Конструкторы и константы сохраняются как символы.
    """
    if isinstance(obj, PropVar):
        shape.append("•")
        return
    if isinstance(obj, Term):
        if obj.is_variable:
            shape.append("v")
        else:
            symbols.add(f"c:{obj.name}")
            shape.append(obj.name)
        return
    if isinstance(obj, Formula):
        tag = type(obj).__name__
        symbols.add(tag)
        shape.append(f"({tag}")
        for f in fields(obj):
            # имена связанных/свободных переменных в полях-строках стираем
            val = getattr(obj, f.name)
            if isinstance(val, str):
                continue
            _walk_parts(val, symbols, shape)
        shape.append(")")
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _walk_parts(item, symbols, shape)
        return
    # str (имена переменных у кванторов/предикатов) и прочее — стираем.


def formula_signature(formula: Formula) -> Tuple[frozenset, str]:
    """(сигнатурные символы, shape-скелет) канонизированной формулы."""
    symbols: Set[str] = set()
    shape: List[str] = []
    _walk_parts(formula, symbols, shape)
    return frozenset(symbols), "".join(shape)


# ── Веса ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UtilityWeights:
    resonance_exact: float = 10.0   # точное попадание в веху
    resonance_shape: float = 3.0    # совпадение формы с вехой
    subsume: float = 8.0            # φ ⊒ веха (веха — инстанция φ)
    goal_step: float = 6.0          # φ = α→β и β ⊒ веха (шаг к цели)
    relevance: float = 2.0          # SInE-близость к невыведенной вехе
    rediscovery: float = 1.0        # log(1+r)
    size: float = 1.0               # −log(size)
    importance: float = 4.0         # выученная I(φ) (importance_v1), если
                                    # модель подключена


# ── Ранкер ────────────────────────────────────────────────────────────────

class UtilityRanker:
    """Ранжирует формулы по полезности U(φ) относительно watchlist'а вех.

    watchlist: последовательность (имя, Formula). Формулы канонизируются
    той же политикой ``structural_formula_form_v1`` (hash-равенство — и есть
    наш матчинг, как в кросс-корпусном протоколе HUBS_AND_FORMULAS §8.4).
    """

    def __init__(
        self,
        watchlist: Sequence[Tuple[str, Formula]] = (),
        weights: Optional[UtilityWeights] = None,
        importance_model=None,
    ):
        self.weights = weights or UtilityWeights()
        self.policy = UTILITY_POLICY
        # Выученная важность I(φ) — объект с .score_formula(formula,
        # rediscovery) -> (0,1) (см. hilbert/importance.py, importance_v1).
        self.importance_model = importance_model

        self._exact: Dict[str, str] = {}      # canonical hash -> имя вехи
        self._shape: Dict[str, str] = {}      # shape -> имя вехи
        self._milestone_symbols: Dict[str, frozenset] = {}  # имя -> символы
        self._milestone_formula: Dict[str, Formula] = {}    # имя -> canonical
        self._hit: Set[str] = set()           # имена выведенных вех

        symbol_counts: Dict[str, int] = {}
        for name, f in watchlist:
            cf, h = canonicalize_with_hash(f)
            syms, shape = formula_signature(cf)
            self._exact[h] = name
            self._shape.setdefault(shape, name)
            self._milestone_symbols[name] = syms
            self._milestone_formula[name] = cf
            for s in syms:
                symbol_counts[s] = symbol_counts.get(s, 0) + 1
        # IDF-вес символа: редкий в корпусе вех — информативный.
        n = max(1, len(self._milestone_symbols))
        self._symbol_weight: Dict[str, float] = {
            s: 1.0 + math.log(n / c) for s, c in symbol_counts.items()
        }

    # — watchlist-прогресс —

    @property
    def watchlist_size(self) -> int:
        return len(self._milestone_symbols)

    @property
    def hits_version(self) -> int:
        """Монотонный счётчик попаданий — ключ инвалидации кэша оценок."""
        return len(self._hit)

    @property
    def hits(self) -> Set[str]:
        return set(self._hit)

    def all_hit(self) -> bool:
        return len(self._hit) == self.watchlist_size and self.watchlist_size > 0

    def check_hit(self, schema_hash: str) -> Optional[str]:
        """Если hash — веха, пометить выведенной и вернуть имя."""
        name = self._exact.get(schema_hash)
        if name is not None and name not in self._hit:
            self._hit.add(name)
            return name
        return None

    # — компоненты U —

    def _relevance(self, syms: frozenset) -> float:
        """max по невыведенным вехам взвешенного Jaccard символов."""
        best = 0.0
        for name, msyms in self._milestone_symbols.items():
            if name in self._hit:
                continue
            inter = syms & msyms
            union = syms | msyms
            if not union:
                continue
            wi = sum(self._symbol_weight.get(s, 1.0) for s in inter)
            wu = sum(self._symbol_weight.get(s, 1.0) for s in union)
            if wu > 0:
                best = max(best, wi / wu)
        return best

    def score(
        self,
        formula: Formula,
        schema_hash: str,
        *,
        rediscovery: int = 0,
    ) -> float:
        """U(φ). Формула предполагается уже канонизированной."""
        w = self.weights
        syms, shape = formula_signature(formula)
        u = 0.0
        if schema_hash in self._exact:
            u += w.resonance_exact
        elif shape in self._shape:
            u += w.resonance_shape
        # Backward-резонанс к невыведенным вехам: φ ⊒ веха (выводит её
        # инстанциацией) и/или консеквент φ ⊒ веха (φ — кандидат последнего
        # MP-шага к цели). По одному бонусу каждого типа.
        if (w.subsume or w.goal_step) and self._milestone_formula:
            # Голый PropVar-консеквент (α→x0) ⊒-покрывает всё — не сигнал.
            is_impl = isinstance(formula, Implies) and not isinstance(
                formula.consequent, PropVar
            )
            subsume_hit = False
            goal_step_hit = False
            for name, mf in self._milestone_formula.items():
                if name in self._hit:
                    continue
                if not subsume_hit and schema_match(formula, mf) is not None:
                    subsume_hit = True
                if (
                    not goal_step_hit
                    and is_impl
                    and schema_match(formula.consequent, mf) is not None
                ):
                    goal_step_hit = True
                if subsume_hit and goal_step_hit:
                    break
            if subsume_hit:
                u += w.subsume
            if goal_step_hit:
                u += w.goal_step
        if w.relevance and self._milestone_symbols:
            u += w.relevance * self._relevance(syms)
        if rediscovery:
            u += w.rediscovery * math.log1p(rediscovery)
        u -= w.size * math.log(max(2, formula.size()))
        if self.importance_model is not None and w.importance:
            u += w.importance * self.importance_model.score_formula(
                formula, rediscovery
            )
        return u

    # — отбор —

    def select(
        self,
        entries: Iterable,
        k: int,
        rediscovery_counts: Optional[Dict[int, int]] = None,
    ) -> List[Tuple[float, object]]:
        """Top-k записей пула по U; детерминированный tie-break по id.

        entries: SchemaEntry-подобные объекты (поля formula, schema_hash, id).
        Возвращает [(score, entry)] по убыванию score.
        """
        rc = rediscovery_counts or {}
        scored = [
            (self.score(e.formula, e.schema_hash, rediscovery=rc.get(e.id, 0)), e)
            for e in entries
        ]
        scored.sort(key=lambda p: (-p[0], p[1].id))
        return scored[: max(0, k)]

    def get_unproved(self) -> List[str]:
        return [n for n in self._milestone_formula if n not in self._hit]

    def metadata(self) -> Dict[str, object]:
        w = self.weights
        return {
            "utility_policy": self.policy,
            "importance_model": (
                getattr(self.importance_model, "policy", "custom")
                if self.importance_model is not None else None
            ),
            "weights": {
                "resonance_exact": w.resonance_exact,
                "resonance_shape": w.resonance_shape,
                "subsume": w.subsume,
                "goal_step": w.goal_step,
                "relevance": w.relevance,
                "rediscovery": w.rediscovery,
                "size": w.size,
                "importance": w.importance,
            },
            "watchlist_size": self.watchlist_size,
        }
