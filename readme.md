# Intelligent Robot Navigation Using Reinforcement Learning in Dynamic Environments

## 1. Project Overview

This project implements an intelligent robot navigation system using **Reinforcement Learning, specifically the Q-Learning algorithm**.

The robot navigates through a grid-based environment containing static and moving obstacles. It learns to reach a specified destination while avoiding collisions through repeated interactions with its environment.

The project also includes an interactive Pygame interface that allows users to create custom environments and observe how the previously trained agent navigates through them.

## 2. Objectives

- Implement Q-Learning for intelligent robot navigation.
- Train an agent across multiple generated environments.
- Enable the robot to navigate around static and moving obstacles.
- Evaluate the trained agent on previously unseen generated scenarios.
- Develop an interactive interface for testing the trained agent in custom environments.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Reinforcement Learning | Learning approach |
| Q-Learning | Learning algorithm |
| Pygame | Interactive environment and visualization |
| NumPy | Numerical operations, where used |
| Matplotlib | Performance graphs |
| CSV | Storing training and evaluation results |
| Pickle | Saving and loading the Q-table |

## 4. Project Structure

```text
intelligent_robot_navigation/
│
├── agents/
│   ├── __init__.py
│   ├── q_learning.py
│   └── state_encoder.py
│
├── environment/
│   ├── __init__.py
│   ├── grid_world.py
│   ├── obstacles.py
│   └── scenario_generator.py
│
├── experiments/
│   ├── __init__.py
│   ├── config.py
│   ├── demo.py
│   ├── evaluate.py
│   ├── evaluate_v2.py
│   └── train.py
│
├── interface/
│   ├── __init__.py
│   ├── app.py
│   └── controls.py
│
├── models/
│   ├── q_table.pkl
│   └── q_table_v2.pkl
│
├── results/
│   ├── graphs/
│   ├── evaluation_results_v2.csv
│   ├── training_results.csv
│   └── training_results_v2.csv
│
├── visualization/
│   ├── __init__.py
│   ├── plots.py
│   └── renderer.py
│
├── main.py
├── requirements.txt
└── README.md
```

## 5. Installation and Setup

### Prerequisites

- Python 3 installed.
- Git (optional, for cloning the repository).
- Required Python libraries.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Intelligent-Robot-Navigation.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Navigate to the project directory

```bash
cd Intelligent-Robot-Navigation
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## 6. Running the Project

Run the following commands from the project's root directory.

### A. Interactive Robot Demonstration

```bash
python -m interface.app
```

This launches the Pygame application, where users can create custom obstacle environments and observe the trained robot navigating through them.

### B. Train the Q-Learning Agent

```bash
python -m experiments.train
```

This starts the training process. The agent learns through interactions with generated environments and updates its Q-table.

The current training configuration uses 5,000 episodes.

After training, the Q-table and training results are saved.

**Note:** Running training again may overwrite the previously saved model and training results.

### C. Evaluate the Trained Agent

```bash
python -m experiments.evaluate_v2
```

This evaluates the saved Q-Learning agent on 500 generated test scenarios without performing additional learning.

### D. Generate Performance Graphs

```bash
python -m visualization.plots
```

This generates graphs representing training and evaluation performance inside the `results/graphs/` directory.

## 7. Interactive Application Controls

The Pygame interface provides the following controls:

| Mouse Control | Function |
|---|---|
| Left Click | Add or remove a static obstacle |
| Right Click | Add a moving obstacle |
| Middle Click | Set the goal position |
| Start Button | Start robot navigation |
| Reset/Edit Button | Reset navigation and modify the environment |

### Environment Elements

- **Blue:** Starting position.
- **Red:** Robot.
- **Green:** Goal position.
- **Black:** Static obstacles.
- **Yellow:** Moving obstacles.

## 8. Q-Learning Algorithm

Q-Learning is a model-free, value-based Reinforcement Learning algorithm.

The agent learns the expected value of taking an action in a particular state and uses these values to select actions.

The Q-value update equation is:

\[
Q(s,a) \leftarrow Q(s,a)+
\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
\]

Where:

- \(Q(s,a)\): Current Q-value.
- \(\alpha\): Learning rate.
- \(r\): Reward received.
- \(\gamma\): Discount factor.
- \(s'\): Next state.
- \(a'\): Possible next action.

The agent uses an epsilon-greedy strategy to balance exploration and exploitation during training.

## 9. Experimental Results

The project was trained for 5,000 episodes and subsequently evaluated on 500 generated test scenarios.

### Training Results

| Metric | Result |
|---|---:|
| Total Episodes | 5,000 |
| Successful Episodes | 3,800 |
| Success Rate | 76.00% |
| Collision Rate | 17.04% |
| Timeout Rate | 6.96% |

### Evaluation Results

| Metric | Result |
|---|---:|
| Total Test Episodes | 500 |
| Successful Episodes | 491 |
| Success Rate | 98.20% |
| Collision Rate | 0.00% |
| Timeout Rate | 1.80% |
| Average Steps | 21.41 |

These are the recorded experimental results from the current implementation. Evaluation scenarios were generated using the same scenario-generation approach as training, but were newly generated test scenarios.

Performance graphs are available in `results/graphs/`.

## 10. Limitations

- The agent uses a compact state representation, which may not distinguish every possible global environment configuration.
- Performance on arbitrary custom environments is not guaranteed.
- The scenario generator checks static path reachability, but does not guarantee a collision-free route around moving obstacles.
- The agent's performance depends on the training environments and learned Q-values.

## 11. Future Scope

- Improve generalization to more complex and arbitrary environments.
- Improve state representation and obstacle awareness.
- Experiment with larger grid environments.
- Improve navigation around multiple moving obstacles.
- Explore more advanced reinforcement learning approaches.

## 12. Author Information

**Project Title:** Intelligent Robot Navigation Using Reinforcement Learning in Dynamic Environments

**Student Name:** [Devanshu Sharma]

**Registration Number:** [23FE10CSE00274]

**Course:** [Reinforecement Learning]

**Department/Institution:** [Department of CSE]

---

**Note:** This project is developed for academic and educational purposes.