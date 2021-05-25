from typing import NamedTuple, List, Optional
import random
import numpy as np
from matplotlib import pyplot as plt, ticker as tck


class PlayerConfig(NamedTuple):
    name: str
    handicap: float = 0.
    use_ifa_mats: bool = False
    use_ifa_nations: bool = False
    nation_blacklist: Optional[str] = None


NORD = 'nordic'
RUS = 'rusviet'
CRIM = 'crimea'
SAX = 'saxony'
POL = 'polania'
nations_standard = [NORD, RUS, CRIM, SAX, POL]
TOG = 'togawa'
ALB = 'albion'
nations_ifa = nations_standard + [TOG, ALB]

AG = 'agricultural'
IND = 'industrial'
PAT = 'patriotic'
MECH = 'mechanical'
ENG = 'engineering'
mats_standard = [AG, IND, PAT, MECH, ENG]
MIL = 'militant'
INN = 'innovative'
mats_ifa = mats_standard + [MIL, INN]


tiers = ['SS', 'S', 'A', 'B', 'C', 'D', 'E']
TIER_LIST_READABLE = dict(
    SS=[(CRIM, PAT), (CRIM, MIL), (CRIM, INN), (RUS, MIL), (RUS, INN)],
    S=[(CRIM, IND), (RUS, IND), (RUS, ENG), (CRIM, MECH), (RUS, MECH), (POL, INN), (SAX, INN)],
    A=[(NORD, IND), (POL, IND), (SAX, IND), (CRIM, ENG), (RUS, PAT), (RUS, AG), (POL, MIL), (NORD, INN)],
    B=[(NORD, ENG), (NORD, PAT), (POL, PAT), (SAX, PAT), (POL, MECH), (SAX, MECH), (POL, AG), (CRIM, AG), (ALB, MIL),
       (SAX, MIL), (TOG, INN), (ALB, INN)],
    C=[(POL, ENG), (ALB, PAT), (TOG, PAT), (NORD, MECH), (NORD, AG), (TOG, AG), (NORD, MIL), (TOG, MIL)],
    D=[(TOG, IND), (ALB, ENG), (SAX, ENG), (TOG, ENG), (ALB, AG), (SAX, AG)],
    E=[(ALB, MECH), (TOG, MECH), (ALB, IND)]
)
banned_tiers = 'SS', 'E'

TIERS_PER_COMBO = {(nation, mat): tier for nation in nations_ifa for mat in mats_ifa for tier in tiers if
                  (nation, mat) in TIER_LIST_READABLE[tier]}


def generate_all_combos(players, nations_pool=None, mats_pool=None):
    if nations_pool is None:
        nations_pool = nations_ifa.copy()
    if mats_pool is None:
        mats_pool = mats_ifa.copy()

    player = players[0]
    viable_nations = [nation for nation in nations_pool if (player.use_ifa_nations or nation in nations_standard and
                                                            nation != player.nation_blacklist)]
    viable_mats = [mat for mat in mats_pool if (player.use_ifa_mats or mat in mats_standard)]
    combos_this_player = [(nation, mat) for nation in viable_nations for mat in viable_mats]

    for combo in combos_this_player:
        if TIERS_PER_COMBO[combo] in banned_tiers:
            continue
        if len(players) == 1:
            yield [combo]
        else:
            nation, mat = combo
            for other_player_combos in generate_all_combos(players=players[1:],
                                                           nations_pool=[n for n in nations_pool if n != nation],
                                                           mats_pool=[m for m in mats_pool if m!= mat]):
                yield [combo] + other_player_combos


tier_idcs = dict(
    D=0, C=1, B=2, A=3, S=4
)
combos_per_tier = [len(TIER_LIST_READABLE[k]) for k in ['D', 'C', 'B', 'A', 'S']]


def weight_per_tier(n, handicap):
    mu = 2 + 1.25 * handicap
    sigma = (3 + 1.5 * np.abs(handicap)) / 2
    return np.exp(-((n - mu) / sigma) ** 2)


def weight_player(combo, player):
    assert len(tier_idcs) == 5  # implicitly assumed in mu, sigma
    weights_per_tier = [weight_per_tier(n, player.handicap) for n in range(len(tier_idcs))]
    normalisation = sum(weights_per_tier[n] * combos_per_tier[n] for n in range(len(tier_idcs)))
    weights_per_tier = [w / normalisation for w in weights_per_tier]
    return weights_per_tier[tier_idcs[TIERS_PER_COMBO[combo]]]


def weight_all_players(combos, players):
    return np.prod([weight_player(c, p) for c, p in zip(combos, players)])


def generate_combos_and_probabilities(players):
    all_combos = list(generate_all_combos(players))
    if len(all_combos) == 0:
        raise RuntimeError
    weights = [weight_all_players(combos, players) for combos in all_combos]
    normalisation = sum(weights)
    return all_combos, [w / normalisation for w in weights]


def pick_boards(player_configs: List[PlayerConfig]):
    # check sanity
    for player in player_configs:
        assert -1 <= player.handicap <= 1
        assert player.nation_blacklist is None or player.nation_blacklist in nations_ifa

    prob_threshold = random.uniform(0., 1.)
    accumulate = 0.
    all_combos, probabilities = generate_combos_and_probabilities(player_configs)
    assert np.allclose(np.sum(probabilities), 1.)
    for combos, prob in zip(all_combos, probabilities):
        accumulate += prob
        if accumulate >= prob_threshold:
            return combos
    raise RuntimeError


def parse_players_command_line():
    num_players = int(input('How many players? '))
    assert 2 <= num_players <= 7

    print()

    names = []
    for n in range(num_players):
        names.append(input(f'Name {n + 1}: '))

    print()

    if input('Any Handicaps (enter to skip)? '):
        print('MIN: -1 (worse combos),   MAX:  +1 (better combos)')
        handicaps = []
        for name in names:
            val = input(f'Handicap for {name}: ')
            if len(val) == 0:
                handicaps.append(0)
            else:
                val = float(val)
                assert -1. <= val <= 1.
                handicaps.append(val)
    else:
        handicaps = [0] * len(names)

    print()

    if input('Any Blacklisting (enter to skip)? '):
        print('Everyone can blacklist one nation')
        blacklists = []
        for name in names:
            val = input(f'Blacklist for {name}: ')
            blacklists.append(None if len(val) == 0 else val)
    else:
        blacklists = [None] * len(names)

    print()

    players = [PlayerConfig(name=n, handicap=h, nation_blacklist=b) for n, h, b in zip(names, handicaps, blacklists)]

    print('I got these players:')
    for player in players:
        msg = f'    {player.name}:  '
        if player.handicap != 0:
            msg += f'handicap={player.handicap}   '
        if player.nation_blacklist is not None:
            msg += f'blacklist={player.nation_blacklist}'
        print(msg)

    if input('Correct? (Enter to continue)'):
        print()
        print()
        return parse_players_command_line()
    else:
        return players


if __name__ == '__main__':
    players = parse_players_command_line()
    # make absolutely sure the system is fair and does not depend on order of players
    random.shuffle(players)
    combos = pick_boards(players)

    print()
    print()
    for player, (nation, mat) in zip(players, combos):
        print(f'    {player.name}: {nation}, {mat}  ({TIERS_PER_COMBO[nation, mat]})')

    # for plotting the probability-per-tier-curves
    # def _normalise(l):
    #     l = list(l)
    #     n = sum(l)
    #     return [x / n for x in l]
    #
    # fig, ax = plt.subplots()
    # ns = list(range(5))
    # for h in [-1, 0, 1]:
    #     ax.plot(ns, _normalise(weight_per_tier(n, h) for n in ns), 'o-', label=f'handicap {h}')
    # plt.xticks([0, 1, 2, 3, 4], ['D', 'C', 'B', 'A', 'S'])
    # ax.yaxis.set_major_formatter(tck.PercentFormatter(1.))
    # fig.legend()
    # plt.show()
