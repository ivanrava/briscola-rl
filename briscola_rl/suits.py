from typing import Union
import numpy as np


class Suit:
    Swords = 1
    Coins = 2
    Cups = 3
    Batons = 4
    __ohe = np.eye(4)

    @classmethod
    def get_seed(cls, i: Union[str, int]):
        assert 0 <= i <= 3, i
        if isinstance(i, str):
            return cls.__dict__[i.capitalize()]
        if i == 0:
            return cls.Swords
        elif i == 1:
            return cls.Coins
        elif i == 2:
            return cls.Cups
        elif i == 3:
            return cls.Batons
        else:
            raise ValueError(f"input {i} should be between [0, 3]")

    @classmethod
    def ohe_repr(cls, suit):
        return cls.__ohe[suit - 1, :]
