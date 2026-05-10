from dataclasses import dataclass, replace
from itertools import permutations

######################
# Game Configuration #
######################


@dataclass(frozen=True)
class Game:
    players: tuple[int, ...]  # the attractiveness values for the players' films
    demand: tuple[int, ...]  # per-week demand
    alphas: tuple[float, ...]  # timing payoffs
    beta: float  # switching cost

    def __post_init__(self) -> None:
        assert len(self.demand) == len(self.alphas)

    @property
    def num_players(self) -> int:
        return len(self.players)

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
class DecisionState:
    week: int
    ordering: tuple[int, ...]  # player ordering
    turn: int  # within player ordering
    releases: tuple[int | None, ...]  # per-played planned or actual release dates
    switches: tuple[int, ...]  # per-player number of release date switches

    def __post_init__(self) -> None:
        num_players = len(self.ordering)
        assert self.turn < num_players
        assert len(self.releases) == num_players
        assert len(self.switches) == num_players

    @property
    def current_player(self) -> int:
        return self.ordering[self.turn]


@dataclass(frozen=True)
class ChanceState:
    week: int
    releases: tuple[int | None, ...]  # current announcement weeks
    switches: tuple[int, ...]  # number of changes


type State = DecisionState | ChanceState


#######################
# Transition Function #
#######################


def initial_state(game: Game) -> State:
    return ChanceState(0, (None,) * game.num_players, (0,) * game.num_players)


def valid_actions(game: Game, state: State) -> tuple[Action, ...]:
    # For ChanceState, the only valid action is to sample (represented as None).
    if isinstance(state, ChanceState):
        return None

    # For DecisionState, valid actions are do nothing (None) or pick a new week (int).
    planned_release = state.releases[state.current_player]
    if state.week > planned_release:
        # If the release already happened, it can't be changed.
        return (None,)
    else:
        # If the release hasn't happened yet, it can be changed.
        weeks = range(state.week, game.num_weeks)
        return (None, *[w for w in weeks if w != planned_release])


def transition(game: Game, state: State, action: Action) -> tuple[State, ...]:
    if isinstance(state, ChanceState):
        # Return all possible turn orderings for the next week.
        possibilities = [
            DecisionState(state.week, tuple(order), 0, state.releases, state.switches)
            for order in permutations(range(game.num_players))
        ]
        return tuple(possibilities)

    # Update the releases and switches based on the action.
    releases = list(state.releases)
    switches = list(state.switches)
    if action is not None:
        if releases[state.current_player] is not None:
            switches[state.current_player] += 1
        releases[state.current_player] = action
    releases = tuple(releases)
    switches = tuple(switches)

    if state.turn == game.num_players - 1:
        # If all players have gone, we must pick a random ordering again.
        return ChanceState(state.week + 1, releases, switches)
    else:
        # Otherwise, we advance to the next player.
        return replace(state, turn=state.turn + 1, releases=releases, switches=switches)


def payoff(game: Game, state: State) -> float:
    pass


if __name__ == "__main__":
    game = Game(
        (10.0, 10.0),  # two equally strong players
        (0.0, 0.0, 5.0, 5.0),  # two weeks with equal demand
        (-0.01, 0.01, 0.01, 0.01),  # best to release after waiting one week
        0.1,  # cost of switching
    )

    a = 1
