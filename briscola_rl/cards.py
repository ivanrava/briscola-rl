from dataclasses import dataclass
from briscola_rl.game_rules import values_points
from briscola_rl.suits import Suit
from random import shuffle


@dataclass()
class Card:
    value: int
    suit: int

    def __post_init__(self):
        assert 0 <= self.value <= 10, self.value
        assert 0 <= self.suit <= 4, self.suit
        self.points = values_points[self.value]
        self.id = self.suit * 10 + self.value

    def vector(self) -> tuple:
        return self.value, self.suit, self.points


NULLCARD_VECTOR = (0, 0, 0)


class Deck:
    __slots__ = ['cards']

    def __init__(self):
        self.cards = self.all_cards()
        shuffle(self.cards)

    @classmethod
    def all_cards(cls):
        return [Card(i % 10 + 1, Suit.get_suit(i // 10)) for i in range(52) if i % 10 + 1 not in [8, 9, 10]]

    def draw(self):
        return self.cards.pop(0)

    def is_empty(self) -> bool:
        return len(self.cards) == 0
