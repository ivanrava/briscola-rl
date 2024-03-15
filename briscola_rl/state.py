from dataclasses import dataclass
from typing import List

from briscola_rl.cards import Card, NULLCARD_VECTOR


def pad_card_vector(lst: list, max_len: int):
    if len(lst) < max_len:
        for _ in range(max_len - len(lst)):
            lst.append(NULLCARD_VECTOR)
    return lst


@dataclass(frozen=True)
class PublicState:
    my_points: int
    other_points: int
    hand: List[Card]
    other_hand_size: int  # TODO: to keep or not to keep?
    remaining_deck_cards: int
    table: List[Card]
    my_taken: List[Card]
    other_taken: List[Card]
    turn: int
    briscola: Card
    my_turn_player: int

    def as_dict(self) -> dict:
        return dict(
            my_points=self.my_points,
            other_points=self.other_points,
            hand_size=len(self.hand),
            other_hand_size=self.other_hand_size,
            remaining_deck_cards=self.remaining_deck_cards,
            hand=pad_card_vector([c.vector() for c in self.hand], 3),
            table=pad_card_vector([c.vector() for c in self.table], 2),
            my_taken=pad_card_vector([c.vector() for c in self.my_taken], 40),
            other_taken=pad_card_vector([c.vector() for c in self.other_taken], 40),
            turn=self.turn,
            briscola=self.briscola.vector(),
            order=self.my_turn_player
        )
