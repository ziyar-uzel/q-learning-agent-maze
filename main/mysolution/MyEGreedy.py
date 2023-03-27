import random


class MyEGreedy:

    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        # TODO to select an action at random in State s
        valid_actions = maze.get_valid_actions(agent)
        return random.choice(valid_actions)

    def get_best_action(self, agent, maze, q_learning):
        # TODO to select the best possible action currently known in State s.
        valid_actions = maze.get_valid_actions(agent)
        current_state = agent.get_state(maze)
        q_vals_for_actions = q_learning.get_action_values(current_state, valid_actions)
        if q_vals_for_actions.count(0 == len(q_vals_for_actions)):
            return self.get_random_action(agent, maze)
        max_index = q_vals_for_actions.index(max(q_vals_for_actions))
        return valid_actions[max_index]

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        exploration_rate = random.uniform(0, 1)
        if exploration_rate < epsilon:
            action = self.get_random_action(agent, maze)
            return action
        else:
            action = self.get_best_action(agent, maze, q_learning)
            return action

