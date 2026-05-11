"""Note: This was coded by hand!"""

from dataclasses import dataclass

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

# None means do nothing, and an int means switching to that release date.
type Action = int | None


@dataclass(frozen=True)
class State:
    turn: int
    releases: tuple[int | None, ...]

    @property
    def num_players(self) -> int:
        return len(self.releases)

    @property
    def current_player(self) -> int:
        return self.turn % self.num_players

    @property
    def week(self) -> int:
        return self.turn // self.num_players


################################
# Transition Function + Reward #
################################


def initial_state(game: Game) -> State:
    return State(0, (None,) * game.num_players)


def valid_actions(game: Game, state: State) -> tuple[Action, ...]:
    # Check whether the game is over.
    if state.week == game.num_weeks:
        return ()

    planned_release = state.releases[state.current_player]
    if planned_release is not None and state.week > planned_release:
        # If the release has already happened, it can't be changed.
        return (None,)
    else:
        # If the release hasn't happened yet, it can be changed.
        weeks = range(state.week, game.num_weeks)
        return (None, *[w for w in weeks if w != planned_release])


def advance(state: State, action: Action) -> State:
    # Update the releases based on the action.
    releases = list(state.releases)
    if action is not None:
        releases[state.current_player] = action
    return State(state.turn + 1, tuple(releases))


type Reward = Float[np.ndarray, " player"]


def compute_reward(
    game: Game,
    old: State,
    new: State,
) -> Reward:
    reward = np.zeros((game.num_players,), dtype=np.float32)

    player = old.current_player
    attractiveness = game.attractivenesses[player]
    old_release = old.releases[player]
    new_release = new.releases[player]

    # Account for timing.
    if new_release is not None and old.week < new_release:
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


################
# Optimization #
################

type Cache = dict[State, tuple[Reward, Action]]


def optimize(
    game: Game,
    state: State,
    cache: Cache,
) -> tuple[Reward, Action]:
    if state in cache:
        return cache[state]

    best_value = None
    best_action = None

    for action in valid_actions(game, state):
        new = advance(state, action)
        reward = compute_reward(game, state, new)
        new_value, _ = optimize(game, new, cache)
        old_value = reward + new_value
        p = state.current_player
        if best_value is None or old_value[p] > best_value[p]:
            best_value = old_value
            best_action = action

    if best_value is None:
        best_value = np.zeros((game.num_players,), dtype=np.float32)

    cache[state] = (best_value, best_action)

    return best_value, best_action


def play(game: Game) -> tuple[Action, ...]:
    cache = {}
    state = initial_state(game)
    actions = []
    while valid_actions(game, state):
        _, best_action = optimize(game, state, cache)
        state = advance(state, best_action)
        actions.append(best_action)
    return tuple(actions)


if __name__ == "__main__":
    game = Game(
        np.array((10.0, 10.0)),
        np.array((0.0, 0.0, 0.0, 4.0, 7.0)),
        np.array((-0.1, -0.1, -0.1, 0.0, 0.0)),
        -0.1,  # cost of switching
    )

    # run this in a debugger, or use the live demo Claude made based on this code
    v, _ = optimize(game, initial_state(game), {})
    actions = play(game)
    print(v)
    print(actions)

    s = initial_state(game)
    for a in actions:
        new = advance(s, a)
        print(compute_reward(game, s, new))
        s = new
