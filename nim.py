import random
import time

"""
Q-Learning算法用于在未知环境中训练一个智能体(agent)做出最优决策。
该算法的核心思想是学习一个价值函数Q(s,a)，其中s表示当前状态，a表示智能体在该状态下采取的行动。
Q(s,a)表示在当前状态下采取行动a所能获得的期望奖励值。Q值越高，则说明该行动对获得最大奖励的贡献越大。

Q-Learning算法用于在未知环境中训练一个智能体(agent)做出最优决策。
该算法的核心思想是学习一个价值函数Q(s,a)，其中s表示当前状态，a表示智能体在该状态下采取的行动。
Q(s,a)表示在当前状态下采取行动a所能获得的期望奖励值。Q值越高，则说明该行动对获得最大奖励的贡献越大。

Q-Learning算法用于在未知环境中训练一个智能体(agent)做出最优决策。
该算法的核心思想是学习一个价值函数Q(s,a)，其中s表示当前状态，a表示智能体在该状态下采取的行动。
Q(s,a)表示在当前状态下采取行动a所能获得的期望奖励值。Q值越高，则说明该行动对获得最大奖励的贡献越大。

Q(s,a) = Q(s,a) + α(r + γ max Q(s',a') - Q(s,a))
其中，Q(s,a)表示在状态s下采取行动a的Q值，α是学习率（控制每次更新的权重），r是执行行动a后，
智能体能够得到的立即奖励，γ是折扣因子（控制未来奖励的权重，表示对未来奖励的重视程度），
s'和a'表示执行当前行动后进入的新状态和新的行动，max(Q(s',a'))表示在下一个状态s'中采取所有可能行动中的最大Q值。

在 Q 学习中，我们尝试为每个（状态、动作）对学习奖励值（数字）。
输掉游戏的动作将获得 -1 的奖励，导致其他玩家输掉游戏的动作将获得 1 的奖励
而导致游戏继续的动作将立即获得 0 的奖励，但会也有一些未来的回报

Nim 游戏的“状态”就是所有桩的当前大小。
例如，状态可能是 [1, 1, 3, 5]，表示第 0 堆中有 1 个对象、第 1 堆中有 1 个对象、第 2 堆中有 3 个对象、第 3 堆中有 5 个对象的状态。
Nim 游戏中的“”将是一对整数 (i, j)，代表从 i 堆中取出 j 个物体的动作。
所以动作 (3, 5) 代表动作“从第 3 堆中拿走 5 个物体”。
将该操作应用于状态 [1, 1, 3, 5] 将导致新状态 [1, 1, 3, 0]（相同的状态，但第 3 堆现在是空的）。

Q(s, a) <- Q(s, a) + alpha * (new value estimate - old value estimate)

新的价值估计表示当前操作收到的奖励与玩家将收到的所有未来奖励的估计之和。
旧的估计值只是 Q(s, a) 的现有值。
通过每次我们的人工智能采取新行动时应用这个公式，随着时间的推移，我们的人工智能将开始学习在任何状态下哪些行动更好

在 Nim 游戏中，我们从一定数量的堆开始，每个堆都有一定数量的物体。
玩家轮流：在玩家的回合中，玩家从任何一个非空堆中移除任何非负数量的物体。
谁拿走最后一个物体，谁就输了。
"""

# Nim()定义了游戏的玩法
class Nim():
    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has
            - `piles`: a list of how many elements remain in each pile
            - `player`: 0 or 1 to indicate which player's turn
            - `winner`: None, 0, or 1 to indicate who the winner is
        """

        # 请注意每个 Nim 游戏都需要跟踪桩列表、当前玩家（0或1）以及游戏的获胜者（如果存在）
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    # available_actions 函数返回某个状态下所有可用操作的集合
    @classmethod
    def available_actions(cls, piles):
        """
        Nim.available_actions(piles) takes a `piles` list as input
        and returns all of the available actions `(i, j)` in that state.

        Action `(i, j)` represents the action of removing `j` items
        from pile `i` (where piles are 0-indexed).
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    # other_player 函数确定给定玩家的对手是谁
    @classmethod
    def other_player(cls, player):
        """
        Nim.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 0 if player == 1 else 1

    # switch_player 将当前玩家更改为对手玩家
    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    # move 对当前状态执行操作并将当前玩家切换为对手玩家
    def move(self, action):
        """
        Make the move `action` for the current player.
        `action` must be a tuple `(i, j)`.
        """
        pile, count = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


# NimAI 类，它定义了我们将学习玩 Nim 的 AI
class NimAI():
    # self.q 字典将通过将（状态、动作）对映射到数值来跟踪 AI 学习到的所有当前 Q 值
    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
        - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4)
        - `action` is a tuple `(i, j)` for an action
        """
        # self.q[(0, 0, 0, 2), (3, 2)] = -1
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    # get_q_value 函数应接受状态和操作作为输入，并返回该状态/操作对的相应 Q 值
    def get_q_value(self, state: list[int], action: tuple[int, int]) -> int:
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        return self.q.get((tuple(state), action), 0)

    # update_q_value函数采用状态state、动作action、现有Q值old_q、当前奖励reward和未来奖励的估计future_rewards
    # 并根据Q学习公式更新状态/动作对的Q值
    # def update_q_value(self, state, action, old_q, reward, future_rewards)
    def update_q_value(self, state: list[int], action: tuple[int], \
                    old_q: float, reward: float, future_rewards: float) -> None:
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        new_q = old_q + self.alpha * (reward + future_rewards - old_q)
        self.q[(tuple(state), action)] = new_q
        return

    # best_future_reward 函数接受一个状态作为输入，并根据 self.q 中的数据返回该状态下任何可用操作的最佳奖励
    def best_future_reward(self, state: list[int]):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """
        return max([self.get_q_value(state, action) \
                    for action in Nim.available_actions(state)], default=0)

    def choose_best_action(self, state: list[int]) -> tuple[int, int]:
        """
        Given a state `state`, return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).
        """
        return max(Nim.available_actions(state), \
                    key=lambda action: self.get_q_value(state, action))

    # Choose_action 函数，它选择在给定状态下采取的操作（贪婪或使用 epsilon-greedy 算法）
    def choose_action(self, state: list[int], epsilon=True) -> tuple[int, int]:
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """
        if not epsilon:
            return self.choose_best_action(state)
        else:
            random_decimal = random.random()
            if random_decimal <= self.epsilon:
                return random.choice(list(Nim.available_actions(state)))
            return self.choose_best_action(state)


def train(n):
    """
    Train an AI by playing `n` games against itself.
    """

    player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:

            # Keep track of current state and action
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")

    # Return the trained AI
    return player


def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = Nim()

    # Game loop
    while True:

        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
