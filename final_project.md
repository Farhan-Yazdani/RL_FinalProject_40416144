# Final Project Document — Reinforcement Learning Course
## Design and Analysis of an Intelligent Agent in a Dynamic Maze

---

## Introduction

This project aims to connect reinforcement learning theory with the design of a working system. You must design a dynamic maze environment, model it as a Markov Decision Process (MDP), and evaluate the performance of three different algorithms under identical conditions. A transfer learning section, a graphical interface, and statistical analysis of results must also be implemented as complementary parts of the project.

The final output must demonstrate that, beyond running code, you understand: the agent's decision-making logic, the difference between model-based and model-free methods, the effect of reward design, the role of exploration vs. exploitation, the concept of eligibility traces, and the limits of knowledge transfer. Simply running a ready-made example or showing a few plots without analysis does not count as a valid response to the project.

## Project Summary

> Build a dynamic maze with a custom map, implement and compare three algorithms — Value Iteration, Q-Learning, and SARSA(λ) — run a limited transfer learning section on Q-Learning, and present results through a graphical interface, heatmaps, visual policies, statistical charts, and an analytical report.

---

## Problem Definition and Maze Environment

The main environment is a 2D maze with a size between 15×15 and 18×18 cells. The agent starts at a specific point, must first obtain a key, then reach an exit door, and finally enter the goal cell.

The environment must include:
- Walls
- Normal cells
- Penalty cells
- Start point
- Key
- Closed door
- Goal

**Structural constraints:**
- At least 15% of cells must be obstacles.
- At least five cells must carry a penalty.

**Transition dynamics:**
At each step, the agent chooses one of four actions: up, down, left, or right.
- With probability 0.8, the chosen action is executed.
- With probability 0.1 (each), the agent deviates to one of the two perpendicular directions (0.2 total).
- Colliding with a wall keeps the agent in its current state and incurs a penalty.

This stochasticity must be reflected in the transition model and in the experiments for all three algorithms.

### State Definition and Preserving the Markov Property

Under a basic definition, the state is not just spatial coordinates, since whether the door is open or closed depends on whether the key has been obtained. Therefore, the minimum state representation is:

```
s = (x, y, k)
```

where `k` is 0 or 1 and indicates whether the key has been collected.

### Mandatory Additional Feature

In addition to the core elements (walls, normal cells, start point, key, exit door, and goal), the maze design must implement **at least one** of the following features:

- Limited energy
- Slippery cell
- Teleporter
- Moving obstacle
- Periodic gate
- Variable goal

**Requirements for this feature:**
- It must have a real effect on the agent's behavior and decision-making process, not just a cosmetic effect.
- It must be clearly displayed in the environment's graphical interface.
- Its mechanics must be explained in the project report.
- The environment's state definition must be adapted to the chosen feature (e.g., for limited energy, remaining energy must be part of the state; for a moving obstacle or variable goal, their current position must be part of the state).
- The state must be defined so that, given the current state and the chosen action, the next-step behavior of the environment can be determined without needing to inspect the history of previous steps.

### Dedicated Seed and Map Generation

The second-to-last digit of the student ID is the base seed for the project. The maze size is also determined from this value.

**Example (student ID 40123456 → seed = 5):**

```python
student_id = '40123456'
base_seed = int(student_id[-2])
maze_size = 15 + (base_seed % 4)
```

General formula:

```
b = int(StudentID[-2])
N = 15 + (b mod 4)
```

**Map validation:**
After generating the map, a deterministic search algorithm such as BFS must confirm that a valid path exists from start to key, and from key to goal. If the map is invalid, it must be fixed or regenerated using a specific, reproducible method. The final map file must be saved so that all algorithms run on exactly the same environment.

---

## MDP Modeling and Reward Function

Before implementing the algorithms, the problem must be formally defined as a Markov Decision Process. The report must precisely describe:
- State space
- Action space
- Transition function
- Reward function
- Discount factor (γ)
- Terminal states
- Agent policy

The MDP definition must be consistent with the environment code, and no variable that affects the future may be omitted from the state representation.

### Two Versions of the Reward Function

**Version 1 — Sparse Reward:**
Most of the reward is given only upon obtaining the key or reaching the goal, and each move carries a small cost.

**Version 2 — Reward Shaping:**
Intermediate behaviors receive appropriate feedback — e.g., moving meaningfully closer to the key, safely passing through a dangerous area, or reducing wasted movement.

The exact reward values are up to you, but you must justify your choices. You must also show whether the shaped reward merely sped up training or also changed the final policy, and whether it introduced unwanted behaviors such as looping movement, collecting side rewards without completing the mission, or excessive avoidance of high-risk cells.

### Minimum Loggable Events

- Normal move
- Wall collision
- Entering a penalty cell
- Obtaining the key
- Attempting to pass through a closed door
- Successfully passing through the door
- Reaching the goal
- Episode ending due to reaching the step cap

### Episode Cap and Termination Condition

Each episode ends when one of the following occurs:
- The agent reaches the goal
- Energy runs out (if limited energy is the chosen feature)
- The number of steps exceeds the defined cap

The suggested cap is **three times** the number of traversable cells. The final chosen value must be recorded in the project's configuration file.

---

## Technical Components Overview

```
Dynamic Maze Environment → [Value Iteration, Q-Learning, SARSA(λ)] → Evaluation & Ablation → Transfer Learning → GUI & Visual Analytics
```

---

## Algorithm 1: Value Iteration

Here, the full environment model — transition probabilities and reward function — is available to the algorithm. Value Iteration must be implemented without using any ready-made library implementation, and used to compute the optimal value function and extract the optimal policy.

$$V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a)\left[R(s, a, s') + \gamma V_k(s')\right]$$

The convergence condition must be based on the maximum change in the value function between two consecutive iterations. The threshold value, discount factor, number of iterations, runtime, and final policy must all be saved.

The policy from this algorithm will serve as the **reference for comparison** with the two model-free methods.

**Requirements:**
- Independent implementation of the Bellman update and greedy policy extraction
- Heatmap of the value function and arrows showing the best action in each state
- Report on number of iterations to convergence and runtime
- Analysis of the effect of at least three different discount factor (γ) values

---

## Algorithm 2: Q-Learning

Q-Learning is implemented as an **off-policy** method. The behavior policy must be ε-greedy, with ε decaying over the course of training. At least two decay schedules (e.g., linear and exponential) must be implemented and compared.

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$$

For proper analysis, the values of α, γ, and ε must be recorded in the configuration file. In addition to the reward curve, the following must be saved per episode:
- Number of steps
- Success rate
- Wall collisions
- Number of entries into penalty cells

At least one real Q-update must be selected from the log file and manually reconstructed in the report.

---

## Algorithm 3: SARSA(λ)

The third algorithm is **on-policy** learning combined with eligibility traces. The trace type can be accumulating or replacing, but the reason for the choice must be explained in the report.

λ must be tested at values 0, 0.3, 0.7, and 0.9.

$$\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t), \qquad Q \leftarrow Q + \alpha \delta_t E$$

$$E_t(s,a) = \gamma \lambda E_{t-1}(s,a) + \mathbf{1}\{s = s_t,\ a = a_t\}$$

You must explain how λ=0 reduces the method to one-step SARSA, and how increasing λ propagates the TD error back to previous states and actions. For at least one short episode, the changes in δ and E over several consecutive steps must be logged and interpreted.

---

## Comparing the Three Algorithms

All three algorithms must be compared on the same map with the same reward definition. The comparison is not limited to final reward, and must also include:
- Runtime
- Number of samples required
- Stability across runs
- Memory usage
- Path quality
- Sensitivity to hyperparameters

To assess policy quality, compute the percentage of states where the greedy action of Q-Learning or SARSA(λ) matches the optimal action from Value Iteration. Differences must be shown on a color-coded map, and at least three example states must be analyzed with respect to the local structure of the environment.

| Criterion | Value Iteration | Q-Learning | SARSA(λ) |
|---|---|---|---|
| Method type | model-based | Model-free, off-policy | Model-free, on-policy |
| Unit of progress | Bellman sweep | Episode | Episode |
| Main output | V and policy | Q and policy | Q, E, and policy |
| Required comparison | Time and convergence | Samples, reward, and stability | Effect of λ and safe behavior |

---

## Transfer Learning Section

This section applies only to Q-Learning. First, the agent is trained in the source environment and its Q-table is saved. Then two target environments are built: one with limited changes, and one with extensive changes.

### Target Environments

**Similar target environment:**
About 15–20% of obstacles are moved, but the start point, key, and goal remain fixed.

**Different target environment:**
At least 35% of obstacles change, the key or goal location moves, and several new penalty cells are added.

Both maps must be validated with BFS to confirm a valid path exists.

### Four Training Scenarios

1. Training from scratch with a zero-initialized Q-table; this is the baseline.
2. Full transfer of the entire Q-table from the source environment to the target environment.
3. Scaled transfer using a factor β to control transfer intensity.
4. Selective transfer; only states whose local neighborhood is unchanged between the two environments are transferred.

$$Q_T^{(0)}(s,a) = \beta\, Q_S(s,a), \qquad \beta \in \{0.25,\ 0.50,\ 0.75\}$$

For each target environment, initial performance, learning speed, and final performance must be reported separately. You must also find at least one example of **negative transfer**, where the transferred knowledge misled the agent into an inappropriate action. The Q-values of that state, the structural change in the environment, and how the behavior was corrected during continued training must all be shown.

> **Analytical expectation**
> The report must distinguish between positive transfer, negative transfer, initial performance, learning speed, and final performance. Statements like "the transfer was good" without a defined metric and numerical evidence are not acceptable.

---

## Graphical Interface and Visualization

The project must have an independent, usable graphical interface. Showing a single static image of the map is not sufficient. The interface can be built with PyQt, Tkinter, Pygame, or a similar tool, and must show the agent's movement step by step.

Walls, normal cells, penalty cells, the key, the door, the goal, the agent, and the chosen additional feature must all have visually distinct markers. Changes in key status, the door opening, obstacle collisions, entering dangerous cells, and successful or unsuccessful episode endings must all be visible in real time.

### Interface Controls

- Selection of algorithm and source/target environment
- Selection of training or evaluation mode
- Start, stop, resume, reset, and re-run
- Animation speed control
- Toggle policy display on/off
- Real-time info display: episode number, step count, reward, ε, key status, and recent success rate

### Visual Outputs

After training ends, the interface or analysis section must be able to display and save as images:
- Value function heatmap
- Final policy arrows
- State visitation count map
- Agent's final path
- Points of policy disagreement
- Difference in Q-values before and after transfer

**Minimum expected content table:**

| Output | Content |
|---|---|
| Value heatmap | V or max-Q value for all valid states |
| Final policy | Arrow for best action, plus markers for terminal states |
| Visitation map | Number of visits to each state during training |
| Policy difference | States agreeing/disagreeing with the reference policy |
| Transfer learning | Difference in Q-values or policy before and after transfer |

### Chart Reporting Rule

Every chart must be followed by at least one analytical paragraph. The analysis must explain the cause of trends, fluctuations, failures, differences between methods, and the limitations of the result; simply restating the axis values does not count as analysis.

---

## Analytical Questions for the Report

The final report, in addition to describing the implementation, must answer the following questions. Answers must reference the code, log files, and actual project results — not just textbook definitions.

1. Define the problem as a complete MDP and explain how your state representation preserves the Markov property.
2. Explain the difference between on-policy and off-policy using the actual behavior of SARSA and Q-Learning near dangerous cells.
3. Why does Value Iteration need a transition model, while the other two algorithms learn without a model of the environment? What are the advantages and limitations of each approach in your project?
4. Which value of λ gave the best balance between learning speed and stability? Explain with numerical evidence.
5. Find three states where the model-free policy differs from the Value Iteration policy, and analyze the likely cause of the discrepancy.
6. In the transfer section, compare the similar and different target environments in terms of initial performance, learning speed, final performance, and negative transfer.

---

## Notes

- Using ready-made RL implementations such as Stable-Baselines or RLlib is **not allowed**.
- Using NumPy, Pandas, Matplotlib, and a GUI library is **allowed**.
- BFS is only for map validation and is not a substitute for the learning agent.
- All charts must be generated from raw data in the repository. Manually editing charts, entering numbers without a corresponding run, discarding failed runs, or reporting only the best seed is not allowed.
