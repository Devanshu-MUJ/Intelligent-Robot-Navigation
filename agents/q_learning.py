
import random
import pickle
import os


class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        action_size=4
    ):

        self.lr = learning_rate
        self.gamma = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.action_size = action_size

        # Dictionary-based Q-table
        self.q_table = {}

    def get_q_values(self, state):

        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_size

        return self.q_table[state]

    
    def choose_action(self, state, training=True):

     # Exploration is allowed only during training
     if training and random.random() < self.epsilon:
        return random.randint(0, self.action_size - 1)

     # During training, create a Q-table entry if needed.
     # During evaluation, use temporary zero values for unseen states.
     if training:
         q_values = self.get_q_values(state)
     else:
         q_values = self.q_table.get(
            state,
            [0.0] * self.action_size
        )

     max_value = max(q_values)

     best_actions = [
        action
        for action, value in enumerate(q_values)
        if value == max_value
     ]

     return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):

        current_q = self.get_q_values(state)[action]

        if done:
            target = reward
        else:
            next_max = max(self.get_q_values(next_state))
            target = reward + self.gamma * next_max

        # Q-Learning update equation
        new_q = current_q + self.lr * (target - current_q)

        self.q_table[state][action] = new_q

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save(self, filename="models/q_table.pkl"):

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        with open(filename, "wb") as file:
            pickle.dump(self.q_table, file)

        print("Q-table saved successfully!")

    def load(self, filename="models/q_table.pkl"):

        with open(filename, "rb") as file:
            self.q_table = pickle.load(file)

        print("Q-table loaded successfully!")
