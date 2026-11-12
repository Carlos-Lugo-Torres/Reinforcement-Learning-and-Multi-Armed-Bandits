"""Agents for Fishing Season (MALIS, Project 1).

This file is a skeleton. Write all your agents in it.

Do NOT modify the classes Agent and ContextualAgent, nor the function play():
they define how every agent interacts with the simulator, and they are used
to run your agents during grading.

How an agent works
------------------
For each season, play(agent, bay) calls:

    agent.reset()                     once, before the first day
    spot = agent.choose()             every day: return a spot, 0 to 4
    agent.update(spot, catch)         every day: the catch you observed

An agent only learns through update(). It must not look at the bay itself.

Contextual agents (Part IV, optional) follow the same pattern, but receive the
morning conditions x (numpy array [wind_kmh, waves_m, tide_m]):

    spot = agent.choose(x)
    agent.update(x, spot, catch)

Example
-------
    from fishing import FishingBay
    from agents import RandomAgent, play

    bay = FishingBay(season=1, seed=0)
    total = play(RandomAgent(seed=0), bay)
"""
import numpy as np

from fishing import ContextualBay, FishingBay

# ---------------------------------------------------------------------------
# Provided: do not modify
# ---------------------------------------------------------------------------


class Agent:
    """Base class for the agents of Parts II and III.

    Parameters
    ----------
    n_spots : int
        Number of fishing spots (5 in the game).
    seed : int or None
        Seed of the agent's own random generator, self.rng. Use self.rng for
        any random decision, so that results can be reproduced.
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self):
        """Return the spot to fish today, an int between 0 and n_spots - 1."""
        raise NotImplementedError

    def update(self, spot, catch):
        """Called after each day with the spot chosen and the catch observed (kg)."""


class ContextualAgent:
    """Base class for the agents of Part IV (contextual bay).

    Same as Agent, except that choose() and update() also receive x, the
    conditions of the day: numpy array [wind_kmh, waves_m, tide_m].
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self, x):
        """Return the spot to fish today given the conditions x."""
        raise NotImplementedError

    def update(self, x, spot, catch):
        """Called after each day with the conditions, the spot chosen and the catch observed."""


def play(agent, bay):
    """Play one full season of bay with agent. Returns the total catch (kg).

    Works with FishingBay and Agent, or with ContextualBay and ContextualAgent.
    The bay is reset first, so the same bay can be played by several agents.
    """
    contextual = isinstance(bay, ContextualBay)
    if contextual and not isinstance(agent, ContextualAgent):
        raise TypeError("a ContextualBay must be played by a ContextualAgent")
    if not contextual and not isinstance(agent, Agent):
        raise TypeError("a FishingBay must be played by an Agent")
    bay.reset()
    agent.reset()
    while not bay.done:
        if contextual:
            x = bay.observe_array()
            spot = int(agent.choose(x))
            catch = bay.fish(spot)
            agent.update(x, spot, catch)
        else:
            spot = int(agent.choose())
            catch = bay.fish(spot)
            agent.update(spot, catch)
    return bay.total


# ---------------------------------------------------------------------------
# Provided example agents (baselines)
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """Chooses a spot uniformly at random every day. Learns nothing."""

    def choose(self):
        return int(self.rng.integers(self.n_spots))


class RandomContextualAgent(ContextualAgent):
    """Ignores the conditions and chooses a spot uniformly at random every day."""

    def choose(self, x):
        return int(self.rng.integers(self.n_spots))


# ---------------------------------------------------------------------------
# Your agents
# ---------------------------------------------------------------------------


class HumanRuleAgent(Agent):
    """Task 4: the best human strategy of our team (Gonzalea's), written as a rule.

    Rule (can be applied by hand)
    -----------------------------
    1. Days 1 to 5 (discovery): fish each spot once, in order 0, 1, 2, 3, 4.
    2. Commit to the spot with the highest average catch (ties: lowest spot number).
    3. Every following day, fish the current spot. A catch is "bad" if it is below
       that spot's own average (before adding the catch). Count the bad catches in
       a row; a catch that is not bad resets the count to 0.
    4. After k bad catches in a row, switch to the spot with the highest average
       among the OTHER spots, and reset the count to 0.

    With k = 6, this rule reproduces every decision Gonzalea made in Seasons 1
    and 3 (on their bays it makes the same choices and gets the same totals).

    Memory
    ------
    sums, counts : total catch and number of visits of each spot (average = sums / counts)
    current      : the spot the agent is committed to (None during discovery)
    bad_streak   : number of bad catches in a row at the current spot

    Parameters
    ----------
    k : int
        Patience: number of bad catches in a row before leaving the current spot.
        Small k reacts fast but leaves good spots after bad luck; large k is
        robust to bad luck but slow to react when the spots change (Season 3).
    """

    def __init__(self, k=6, n_spots=5, seed=None):
        super().__init__(n_spots, seed)
        self.k = k

    def reset(self):
        self.sums = np.zeros(self.n_spots)
        self.counts = np.zeros(self.n_spots, dtype=int)
        self.current = None
        self.bad_streak = 0

    def choose(self):
        # Discovery: go to the first spot not visited yet (so spots 0, 1, 2, 3, 4)
        unvisited = np.flatnonzero(self.counts == 0)
        if len(unvisited) > 0:
            return int(unvisited[0])
        # End of discovery: commit to the spot with the best average
        if self.current is None:
            self.current = int(np.argmax(self.sums / self.counts))
        return self.current

    def update(self, spot, catch):
        # Is it a bad catch at the current spot? (compared with its average before this catch)
        if spot == self.current:
            average = self.sums[spot] / self.counts[spot]
            if catch < average:
                self.bad_streak += 1
            else:
                self.bad_streak = 0

        # Add the catch to the spot's statistics
        self.sums[spot] += catch
        self.counts[spot] += 1

        # Out of patience: switch to the best of the other spots
        if self.current is not None and self.bad_streak >= self.k:
            averages = self.sums / self.counts
            averages[self.current] = -np.inf  # exclude the current spot
            self.current = int(np.argmax(averages))
            self.bad_streak = 0


# Task 4: add below at least two agents implementing methods from the reading
# (Sutton and Barto, Chapter 2), as subclasses of Agent. Document each class:
# what it keeps in memory, how it chooses, what its parameters mean.
class EpsilonGreedyAlphaAgent(Agent):
    """
    Parameters:
    epsilon : float
        Probability of choosing a random spot (exploration).
    alpha : float
        Step-size parameter for updating the estimated values.

    Memory:
    Q: estimated average catch for each spot. 
    N: number of times each spot has been fished.

    How it chooses:
    Spots never fished are tried first, once each.
    Then, each day, with probability epsilon it fishes a spot chosen uniformly at random, otherwise it fishes the spot with the highest Q.
    """

    def __init__(self, epsilon=0.1, alpha=0.5, n_spots=5, seed=None):
        super().__init__(n_spots, seed)
        self.epsilon = epsilon
        self.alpha = alpha

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""
        self.Q = np.zeros(self.n_spots)
        self.N = np.zeros(self.n_spots)

    def choose(self):
        """Return the spot to fish today, depending on the exploration strategy"""
        untried = np.flatnonzero(self.N == 0)
        if len(untried) > 0:
            return int(untried[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_spots))
        return int(np.argmax(self.Q))

    def update(self, spot, catch):
        """Called after each day with the spot chosen and the catch observed."""
        self.N[spot] += 1
        if self.N[spot] == 1:
            step = 1.0
        elif self.alpha is None:
            step = 1.0 / self.N[spot]
        else:
            step = self.alpha
        self.Q[spot] += step * (catch - self.Q[spot])

class UCBAgent(Agent):
    """
    Parameters:
    c : float
        Exploration parameter for the UCB algorithm.

    Memory:
    Q: estimated average catch for each spot. 
    N: number of times each spot has been fished.
    t: number of days played in the season.

    How it chooses:
    Spots never fished are tried first, once each.
    Then, each day, it chooses the spot with the highest upper confidence bound (UCB).
    """

    def __init__(self, c=1.0, n_spots=5, seed=None):
        super().__init__(n_spots, seed)
        self.c = c

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""
        self.Q = np.zeros(self.n_spots)
        self.N = np.zeros(self.n_spots)
        self.t = 0

    def choose(self):
        """Return the spot to fish today, depending on the UCB strategy"""
        self.t += 1
        untried = np.flatnonzero(self.N == 0)
        if len(untried) > 0:
            return int(untried[0])
        bonus = self.c * np.sqrt(np.log(self.t) / self.N)
        return int(np.argmax(self.Q + bonus))

    def update(self, spot, catch):
        """Called after each day with the spot chosen and the catch observed."""
        self.N[spot] += 1
        self.Q[spot] += (catch - self.Q[spot]) / self.N[spot]

# Task 6 (optional): add your contextual agents below, as subclasses of
# ContextualAgent.

class IgnoreConditions(ContextualAgent):
    """Adapter to allow a standard Agent to play in a ContextualBay.
    It ignores the morning conditions 'x' and passes only the required arguments to the underlying base agent.
    With this class we can use any of the agents in a ContextualBay, even if they don't use the context.
    This is useful to compare the agent without context.
    """
    def __init__(self, base_agent):
        super().__init__(base_agent.n_spots)
        self.base_agent = base_agent

    def reset(self):
        self.base_agent.reset()

    def choose(self, x):
        return self.base_agent.choose()

    def update(self, x, spot, catch):
        self.base_agent.update(spot, catch)


class LinearRegressionAgent(ContextualAgent):
    """Contextual Agent using Ordinary Least Squares (Linear Regression).

    Memory:
        X, Y : historical contexts (features) and catches for each spot.
        t    : number of days played in the season.

    How it chooses:
        1. Discovery phase (first m * n_spots days): fishes each spot 'm' times round-robin to gather a solid baseline of weather data.
        2. Exploitation: fits an exact linear regression model (Least Squares) for each spot using its historical data. Predicts the catch for the current conditions 'x' and chooses the spot with the highest expected catch.

    Parameters:
        m : int
            Number of forced discovery days per spot before relying on the model.
    """
    def __init__(self, m=3, n_spots=5, seed=None):
        super().__init__(n_spots, seed)
        self.m = m

    def reset(self):
        self.t = 0
        # Create a separate list for each spot to store history
        self.X = [[] for _ in range(self.n_spots)]
        self.Y = [[] for _ in range(self.n_spots)]

    def choose(self, x):
        # Discovery phase: ensure each spot gets exactly m visits early on
        if self.t < self.m * self.n_spots:
            return self.t % self.n_spots
        
        # Add a bias term (1.0) to the context array
        features = np.append(x, 1.0)
        predictions = np.zeros(self.n_spots)
        
        for spot in range(self.n_spots):
            # Solve theta using exact least squares: X * theta = Y
            theta, _, _, _ = np.linalg.lstsq(self.X[spot], self.Y[spot], rcond=None)
            predictions[spot] = np.dot(features, theta)
            
        return int(np.argmax(predictions))

    def update(self, x, spot, catch):
        self.t += 1
        features = np.append(x, 1.0)
        self.X[spot].append(features)
        self.Y[spot].append(catch)