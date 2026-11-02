
from agents.q_learning import QLearningAgent

agent = QLearningAgent()

agent.load("models/q_table_v2.pkl")

# Record the original Q-table size
original_size = len(agent.q_table)

# Test an unfamiliar state
unseen_state = (99, 99, 99, 99, 99, 99, 99, 99, 99, 99)

# Select an action without training
action = agent.choose_action(
    unseen_state,
    training=False
)

# Check whether the Q-table changed
new_size = len(agent.q_table)

print("Original Q-table size:", original_size)
print("New Q-table size:", new_size)
print("Selected action:", action)

if original_size == new_size:
    print("SUCCESS: Q-table remained unchanged!")
else:
    print("ERROR: Q-table was modified!")
