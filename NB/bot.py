from itertools import combinations
from time import perf_counter
from typing import Any, Iterable, Sequence

from stellatro_common import CardModel, GameState, JokerModel, PlayerTurn
from stellatro_game import Card, Game, JOKER_HAND_SIZE, PLAYER_CARDS, Suit
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker


_JOKER_NAME_TO_CLASS = {cls.name: cls for cls in ALL_JOKER_CLASSES}
_COMBOS_BY_HAND_SIZE = {
    size: list(combinations(range(size), 5))
    for size in range(5, PLAYER_CARDS + 1)
}

STELLA_GENERATORS = {
    "Wish Upon a Star",
    "Binary Star",
    "Pips",
    "Report Card",
    "Star Plasma",
    "Starjack",
    "Thrice Twice",
    "Fallen Star",
    "Star Fish",
    "Branch Out",
}
STELLA_PAYOFFS = {
    "Stargazing",
    "Boiling Point",
    "Galaxy",
    "Starcorn",
    "Supernova",
    "Snowball",
    "Constellation",
}
RETRIGGER_JOKERS = {
    "Sock and Buskin",
    "Seltzer",
    "Last Lecture",
    "Encore",
    "Stargazing",
}
FACE_CARD_JOKERS = {
    "Sock and Buskin",
    "PhotoGraph Joker",
    "Scary Face Joker",
    "Mirror",
    "Spotlight",
    "Starjack",
}
PAIR_JOKERS = {
    "Jolly Joker",
    "Sly Joker",
    "Cheeky Joker",
    "Jovial Joker",
    "The Duo",
    "The Trio",
    "Thrice Twice",
    "Star Fish",
}
FLUSH_JOKERS = {
    "Daring Joker",
    "Vibrant Joker",
    "Diamond Joker",
    "Heart Joker",
    "Club Joker",
    "Spade Joker",
    "Sun God",
    "The Tribe",
    "Encore",
    "Arrowhead",
}
STRAIGHT_JOKERS = {
    "Witty Joker",
    "Lively Joker",
    "The Order",
}

CardSpec = tuple[int, tuple[str, ...], int]
PreparedCombo = tuple[tuple[int, ...], tuple[CardSpec, ...]]


def _card_specs(cards: Sequence[CardModel]) -> tuple[CardSpec, ...]:
    return tuple(
        (card.rank, tuple(card.suits), card.num_triggers)
        for card in cards
    )


def _joker_names(jokers: Iterable[JokerModel]) -> tuple[str, ...]:
    return tuple(joker.name for joker in jokers)


def _prepared_combos(hand_specs: tuple[CardSpec, ...]) -> tuple[PreparedCombo, ...]:
    combos = _COMBOS_BY_HAND_SIZE.get(
        len(hand_specs),
        list(combinations(range(len(hand_specs)), 5)),
    )
    return tuple(
        (combo, tuple(hand_specs[index] for index in combo))
        for combo in combos
    )


def _active_hand_and_jokers(
    state: GameState,
) -> tuple[list[CardModel], list[JokerModel], list[CardModel], list[JokerModel]]:
    if state.current_turn == PlayerTurn.PLAYER1:
        return (
            state.player1_hand,
            state.player1_jokers,
            state.player2_hand,
            state.player2_jokers,
        )
    return (
        state.player2_hand,
        state.player2_jokers,
        state.player1_hand,
        state.player1_jokers,
    )


def _same_family(name_a: str, name_b: str, family: set[str]) -> bool:
    return name_a in family and name_b in family


def _partner_priority(
    candidate: JokerModel,
    partner: JokerModel,
    partner_immediate_score: float,
) -> float:
    priority = partner_immediate_score
    candidate_name = candidate.name
    partner_name = partner.name

    if (
        candidate_name in STELLA_GENERATORS
        and partner_name in STELLA_PAYOFFS
        or candidate_name in STELLA_PAYOFFS
        and partner_name in STELLA_GENERATORS
    ):
        priority += 1_000_000_000
    if (
        candidate_name in RETRIGGER_JOKERS
        and partner_name == "Jam Session"
        or candidate_name == "Jam Session"
        and partner_name in RETRIGGER_JOKERS
    ):
        priority += 900_000_000
    if _same_family(candidate_name, partner_name, FACE_CARD_JOKERS):
        priority += 700_000_000
    if _same_family(candidate_name, partner_name, FLUSH_JOKERS):
        priority += 600_000_000
    if _same_family(candidate_name, partner_name, PAIR_JOKERS):
        priority += 500_000_000
    if _same_family(candidate_name, partner_name, STRAIGHT_JOKERS):
        priority += 400_000_000
    return priority


class Bot:
    DRAFT_TIME_LIMIT_SECONDS = 0.180
    COMBO_TOP_K = 7
    PARTNER_TOP_K = 9
    FUTURE_COMBO_WEIGHT = 0.20
    DENIAL_WEIGHT = 0.35
    DENIAL_TOP_K = 10
    MAX_SCORE_CACHE_SIZE = 50_000
    MAX_BEST_HAND_CACHE_SIZE = 10_000

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._game = Game(verbose=False)
        self._score_cache: dict[tuple[tuple[CardSpec, ...], tuple[str, ...]], float] = {}
        self._best_hand_cache: dict[
            tuple[tuple[CardSpec, ...], tuple[str, ...]],
            tuple[float, tuple[int, ...]],
        ] = {}
        self._joker_class_cache: dict[tuple[str, ...], tuple[Any, ...]] = {}

    def _cards_from_specs(self, specs: Sequence[CardSpec]) -> list[Card]:
        cards = []
        for rank, suit_names, num_triggers in specs:
            first_suit = suit_names[0] if suit_names else Suit.SPADE.value
            card = Card(rank, Suit(first_suit))
            for suit_name in suit_names[1:]:
                card.add_suit(Suit(suit_name))
            card.num_triggers = num_triggers
            cards.append(card)
        return cards

    def _score_specs(
        self,
        card_specs: tuple[CardSpec, ...],
        joker_names: tuple[str, ...],
    ) -> float:
        key = (card_specs, joker_names)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached

        cards = self._cards_from_specs(card_specs)
        joker_classes = self._joker_class_cache.get(joker_names)
        if joker_classes is None:
            joker_classes = tuple(
                _JOKER_NAME_TO_CLASS.get(name, RegularJoker)
                for name in joker_names
            )
            if len(self._joker_class_cache) >= self.MAX_BEST_HAND_CACHE_SIZE:
                self._joker_class_cache.clear()
            self._joker_class_cache[joker_names] = joker_classes

        jokers = [joker_class() for joker_class in joker_classes]
        try:
            score = float(self._game.evaluate_hand(cards, jokers))
        except Exception:
            score = -1.0

        if len(self._score_cache) >= self.MAX_SCORE_CACHE_SIZE:
            self._score_cache.clear()
        self._score_cache[key] = score
        return score

    def _best_prepared_hand(
        self,
        hand_specs: tuple[CardSpec, ...],
        prepared_combos: tuple[PreparedCombo, ...],
        joker_names: tuple[str, ...],
    ) -> tuple[float, list[int]]:
        key = (hand_specs, joker_names)
        cached = self._best_hand_cache.get(key)
        if cached is not None:
            score, indices = cached
            return score, list(indices)

        best_score = -1.0
        best_indices = (
            list(prepared_combos[0][0])
            if prepared_combos
            else list(range(min(5, len(hand_specs))))
        )
        for combo, selected_specs in prepared_combos:
            score = self._score_specs(selected_specs, joker_names)
            if score > best_score:
                best_score = score
                best_indices = list(combo)

        if len(self._best_hand_cache) >= self.MAX_BEST_HAND_CACHE_SIZE:
            self._best_hand_cache.clear()
        self._best_hand_cache[key] = (best_score, tuple(best_indices))
        return best_score, best_indices

    def _best_hand(
        self,
        hand: Sequence[CardModel],
        jokers: Sequence[JokerModel] | tuple[str, ...],
    ) -> tuple[float, list[int]]:
        hand_specs = _card_specs(hand)
        joker_names = jokers if isinstance(jokers, tuple) else _joker_names(jokers)
        return self._best_prepared_hand(
            hand_specs,
            _prepared_combos(hand_specs),
            joker_names,
        )

    def pick_joker(self, state: GameState) -> int:
        joker_pool = state.joker_pool
        if not joker_pool:
            return 0

        start_time = perf_counter()
        my_hand, my_jokers, opponent_hand, opponent_jokers = _active_hand_and_jokers(state)
        my_hand_specs = _card_specs(my_hand)
        opponent_hand_specs = _card_specs(opponent_hand)
        my_prepared_combos = _prepared_combos(my_hand_specs)
        opponent_prepared_combos = _prepared_combos(opponent_hand_specs)
        candidate_scores: list[tuple[float, int, tuple[str, ...]]] = []
        my_joker_names = _joker_names(my_jokers)
        opponent_joker_names = _joker_names(opponent_jokers)
        opponent_base_score: float | None = None

        for index, joker in enumerate(joker_pool):
            candidate_jokers = my_joker_names + (joker.name,)
            score, _ = self._best_prepared_hand(
                my_hand_specs,
                my_prepared_combos,
                candidate_jokers,
            )
            candidate_scores.append((score, index, candidate_jokers))
            if perf_counter() - start_time > self.DRAFT_TIME_LIMIT_SECONDS:
                break

        if not candidate_scores:
            return 0

        candidate_values = {
            index: score
            for score, index, _ in candidate_scores
        }
        immediate_score_by_index = dict(candidate_values)
        opponent_reply_order: tuple[int, ...] = ()
        if len(opponent_jokers) < JOKER_HAND_SIZE:
            opponent_base_score, _ = self._best_prepared_hand(
                opponent_hand_specs,
                opponent_prepared_combos,
                opponent_joker_names,
            )
            reply_scores: list[tuple[float, int]] = []
            for reply_index, reply_joker in enumerate(joker_pool):
                if perf_counter() - start_time > self.DRAFT_TIME_LIMIT_SECONDS:
                    break
                reply_score, _ = self._best_prepared_hand(
                    opponent_hand_specs,
                    opponent_prepared_combos,
                    opponent_joker_names + (reply_joker.name,),
                )
                reply_scores.append((reply_score - opponent_base_score, reply_index))
            opponent_reply_order = tuple(
                index for _, index in sorted(reply_scores, reverse=True)
            )

        can_plan_future_combo = (
            len(my_jokers) + 1 < JOKER_HAND_SIZE
            and len(joker_pool) > 1
            and perf_counter() - start_time <= self.DRAFT_TIME_LIMIT_SECONDS
        )
        if can_plan_future_combo:
            ranked_for_combo = sorted(candidate_scores, reverse=True)[: self.COMBO_TOP_K]
            for my_score, index, candidate_jokers in ranked_for_combo:
                if perf_counter() - start_time > self.DRAFT_TIME_LIMIT_SECONDS:
                    break

                candidate = joker_pool[index]
                partner_options = [
                    (
                        _partner_priority(
                            candidate,
                            partner,
                            immediate_score_by_index.get(partner_index, 0.0),
                        ),
                        partner_index,
                        partner,
                    )
                    for partner_index, partner in enumerate(joker_pool)
                    if partner_index != index
                ]

                likely_opponent_replies = set(
                    tuple(
                        reply_index
                        for reply_index in opponent_reply_order
                        if reply_index != index
                    )[:1]
                )
                for _, partner_index, partner in sorted(partner_options, reverse=True)[: self.PARTNER_TOP_K]:
                    if perf_counter() - start_time > self.DRAFT_TIME_LIMIT_SECONDS:
                        break
                    if partner_index in likely_opponent_replies:
                        continue
                    future_score, _ = self._best_prepared_hand(
                        my_hand_specs,
                        my_prepared_combos,
                        candidate_jokers + (partner.name,),
                    )
                    future_gain = max(0.0, future_score - my_score)
                    value = my_score + self.FUTURE_COMBO_WEIGHT * future_gain
                    if value > candidate_values[index]:
                        candidate_values[index] = value

        best_index, _ = max(candidate_values.items(), key=lambda item: item[1])

        has_opponent_pick_after_us = (
            len(joker_pool) > 1
            and len(opponent_jokers) < JOKER_HAND_SIZE
            and perf_counter() - start_time <= self.DRAFT_TIME_LIMIT_SECONDS
        )
        if not has_opponent_pick_after_us:
            return best_index

        if opponent_base_score is None:
            opponent_base_score, _ = self._best_prepared_hand(
                opponent_hand_specs,
                opponent_prepared_combos,
                opponent_joker_names,
            )
        ranked_candidates = sorted(
            ((value, index) for index, value in candidate_values.items()),
            reverse=True,
        )[: self.DENIAL_TOP_K]

        best_value = float("-inf")
        fallback_index = best_index
        for my_score, index in ranked_candidates:
            if perf_counter() - start_time > self.DRAFT_TIME_LIMIT_SECONDS:
                break

            candidate = joker_pool[index]
            opponent_score, _ = self._best_prepared_hand(
                opponent_hand_specs,
                opponent_prepared_combos,
                opponent_joker_names + (candidate.name,),
            )
            opponent_gain = max(0.0, opponent_score - opponent_base_score)
            value = my_score + self.DENIAL_WEIGHT * opponent_gain
            if value > best_value:
                best_value = value
                best_index = index

        return best_index if best_value > float("-inf") else fallback_index

    def pick_hand(self, state: GameState) -> list[int]:
        my_hand, my_jokers, _, _ = _active_hand_and_jokers(state)
        _, indices = self._best_hand(my_hand, my_jokers)
        return indices
