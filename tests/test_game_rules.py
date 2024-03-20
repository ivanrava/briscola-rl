from briscola_rl import game_rules
from briscola_rl.cards import Card


def test_win_case():
    table = [Card(6, 4), Card(2, 4)]
    assert game_rules.select_winner(table, briscola=Card(2, 3)) == 0