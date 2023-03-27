from main.QLearning import QLearning


class MyQLearning(QLearning):

    def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
        current_q = self.get_q(state, action)
        highest_q_next_step = max(self.get_action_values(state_next, possible_actions))
        new_q = current_q + alfa * (r + gamma * highest_q_next_step-current_q)
        self.set_q(state, action, new_q)
        return new_q
