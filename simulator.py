from dataclasses import dataclass, replace
from itertools import permutations

import numpy as np
from jaxtyping import Float

######################
# Game Configuration #
######################


@dataclass(frozen=True)
class Game:
    attractivenesses: Float[np.ndarray, " player"]
    demand: Float[np.ndarray, " week"]
    alphas: Float[np.ndarray, " week"]
    beta: float

    @property
    def num_players(self) -> int:
        return len(self.attractivenesses)

    @property
    def num_weeks(self) -> int:
        return len(self.demand)


################################
# State and Action Definitions #
################################

# For ChanceState, None is the only valid action. It essentially means "pick among the
# possible states at random." For DecisionState, None means do nothing, and an int means
# switching to that release date.
type Action = int | None


@dataclass(frozen=True)
class ChanceState:
    week: int  # for week == num_weeks, this is a terminal state
    releases: tuple[int | None, ...]


@dataclass(frozen=True)
class DecisionState:
    week: int
    ordering: tuple[int, ...]
    turn: int
    releases: tuple[int | None, ...]

    def __post_init__(self) -> None:
        num_players = len(self.ordering)
        assert self.turn < num_players
        assert len(self.releases) == num_players

    @property
    def current_player(self) -> int:
        return self.ordering[self.turn]


type State = ChanceState | DecisionState


################################
# Transition Function + Reward #
################################


def initial_state(game: Game) -> State:
    return ChanceState(0, (None,) * game.num_players)


def valid_actions(game: Game, state: State) -> tuple[Action, ...]:
    # For ChanceState, the only valid action is to sample (represented as None).
    if isinstance(state, ChanceState):
        return (None,)

    # For DecisionState, valid actions are do nothing (None) or pick a new week (int).
    planned_release = state.releases[state.current_player]
    if planned_release is not None and state.week > planned_release:
        return (None,)
    else:
        # If the release hasn't happened yet, it can be changed.
        weeks = range(state.week, game.num_weeks)
        return (None, *[w for w in weeks if w != planned_release])


def advance(game: Game, state: State, action: Action) -> tuple[State, ...]:
    if isinstance(state, ChanceState):
        # Return all possible turn orderings for the next week.
        possibilities = [
            DecisionState(state.week, tuple(order), 0, state.releases)
            for order in permutations(range(game.num_players))
        ]
        return tuple(possibilities)

    # Update the releases and switches based on the action.
    releases = list(state.releases)
    if action is not None:
        releases[state.current_player] = action
    releases = tuple(releases)

    if state.turn == game.num_players - 1:
        # If all players have gone, we move to the next week.
        return (ChanceState(state.week + 1, releases),)
    else:
        # Otherwise, we advance to the next player.
        return (replace(state, turn=state.turn + 1, releases=releases),)


def compute_reward(
    game: Game,
    old: State,
    new: State,
) -> Float[np.ndarray, " player"]:
    reward = np.zeros((game.num_players,), dtype=np.float32)

    # ChanceState does not create rewards.
    if isinstance(old, ChanceState):
        return reward

    player = old.current_player
    attractiveness = game.attractivenesses[player]
    old_release = old.releases[player]
    new_release = new.releases[player]

    # Account for timing.
    if new_release is not None:
        reward[player] += game.alphas[old.week] * attractiveness

    # Account for switching.
    if old_release is not None and old_release != new_release:
        reward[player] += game.beta * attractiveness

    # Account for releasing.
    if old.week != new.week:
        release_reward = np.zeros_like(reward)
        for i in range(game.num_players):
            if new.releases[i] == old.week:
                release_reward[i] += game.attractivenesses[i]
        total = sum(release_reward)
        demand = game.demand[old.week]
        if total > demand:
            release_reward *= demand / total
        reward += release_reward

    return reward


if __name__ == "__main__":
    game = Game(
        np.array((10.0, 10.0)),  # two equally strong players
        np.array((0.0, 0.0, 5.0, 5.0)),  # two weeks with equal demand
        np.array((-0.01, 0.01, 0.01, 0.01)),  # best to release after waiting one week
        -0.1,  # cost of switching
    )
    old = initial_state(game)
    new = advance(game, old, valid_actions(game, old)[0])[0]
    reward = compute_reward(game, old, new)
    old = new
    new = advance(game, old, 0)[0]
    reward = compute_reward(game, old, new)

    a = 1
