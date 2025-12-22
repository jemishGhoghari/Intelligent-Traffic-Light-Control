# DQN Traffic Light Control

## 🚦 Introduction

Urban traffic congestion is a growing challenge in cities worldwide, leading to increased travel times, fuel consumption, and air pollution. Traditional traffic light systems use fixed timing patterns that cannot adapt to real-time traffic conditions, often resulting in unnecessary delays and long queues at intersections.

### Motivation

This project addresses the limitations of fixed-timing traffic signals by implementing an **intelligent, adaptive traffic light control system** using Deep Reinforcement Learning (DRL). Instead of pre-programmed timing patterns, our system learns optimal signal control policies by observing traffic patterns and making decisions that minimize congestion in real-time.

### Goal

The primary goal is to develop a **Deep Q-Network (DQN) agent** that can:

- Learn optimal traffic light phase switching strategies through interaction with realistic traffic simulations
- Reduce average vehicle waiting times and queue lengths at intersections
- Improve overall traffic flow and vehicle throughput
- Outperform traditional fixed-timing control strategies

This project uses the **SUMO (Simulation of Urban MObility)** traffic simulator with real-world traffic data from Ingolstadt, Germany, providing a realistic testbed for training and evaluating the adaptive control system.

---

## 📋 Prerequisites

Before getting started, ensure you have:

1. **Python 3.8+** installed
2. **SUMO** (Simulation of Urban MObility) installed
3. **Git** for cloning repositories

---

## 🛠️ Installation

Open a fresh terminal and execute the following commands to setup SUMO simulation and repository:

### Step 1: Install SUMO Simulator

**Ubuntu/Linux:**

```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

**macOS:**

```bash
brew install sumo
```

**Windows:**

- Download from [SUMO official website](https://www.eclipse.org/sumo/)
- Install and add SUMO to your system PATH

### Step 2: Clone Repository with Submodules

```bash
git clone https://github.com/jemishGhoghari/Intelligent-Traffic-Light-Control.git
cd Intelligent-Traffic-Light-Control
```

### Step 3: Update Submodules

```bash
git submodule update --init --recursive
```

This will automatically clone the Ingolstadt SUMO network (included as a submodule) which provides the real-world traffic network configuration.

### Step 4: Create Python Virtual Environment

```bash
python3 -m venv venv
```

### Step 5: Activate Virtual Environment

**Linux/macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### Step 6: Install Dependencies

```bash
pip3 install traci libsumo torch numpy matplotlib pyyaml
```

---

## 🚀 Quick Start

### Training the Model

Train the DQN agent from scratch:

```bash
python train_agent.py
```

**What happens:**

- The agent learns to control traffic lights through trial and error
- Training runs for 200 episodes (configurable in `settings/training_settings.yaml`)
- Progress is logged every 50 steps
- Checkpoints are saved every 10 episodes in `training_outputs/`
- Final trained model is saved as `training_outputs/dqn_sumo_policy_final.pt`

**Expected Duration:** 3-12 hours depending on your hardware (GPU recommended)

**Training Configuration:**
Edit `settings/training_settings.yaml` to adjust:

- `total_episodes`: Number of training episodes (default: 200)
- `use_gui`: Set to `true` to visualize training (slower)
- `reward_type`: Choose between `"queue"`, `"waiting_time"`, or `"combined"`

### Testing the Trained Model

Test your trained model with real-time visualization:

```bash
python test_agent.py --model training_outputs/dqn_sumo_policy_final.pt
```

**What you'll see:**

- SUMO GUI showing live traffic simulation
- Real-time plots of queue length, vehicle count, speed, and rewards
- Console output showing the agent's decisions and performance metrics

**Options:**

```bash
# Test without SUMO GUI
python test_agent.py --model training_outputs/dqn_sumo_policy_final.pt --no-gui

# Test without live plots
python test_agent.py --model training_outputs/dqn_sumo_policy_final.pt --no-viz

# Run for 500 steps only
python test_agent.py --model training_outputs/dqn_sumo_policy_final.pt --steps 500
```

### Evaluating Performance

Compare DQN performance against fixed-timing baseline:

```bash
python agent_evaluation.py
```

**What this does:**

- Runs your trained DQN model for 5 episodes
- Runs traditional fixed-timing control (30s green phases) for 5 episodes
- Generates comparison plots in `testing_outputs/`
- Calculates percentage improvements in key metrics

**Results saved:**

- `evaluation_results.json`: Numerical comparison
- `queue_length_comparison.png`: Queue length over time
- `waiting_time_comparison.png`: Waiting time comparison
- `speed_comparison.png`: Average speed comparison
- `summary_comparison.png`: Overall performance metrics

---

## 📁 Project Structure

```
dqn-traffic-light-control/
│
├── sumo_ingolstadt/              # SUMO network (clone from github)
│   └── simulation/
│       └── 24h_bicycle_sim.sumocfg
│
├── settings/
│   └── training_settings.yaml    # Configuration file
│
├── dqn_model.py                  # Neural network architecture
├── sumo_simulation_env.py        # SUMO environment wrapper
├── train_agent.py                # Training script
├── test_agent.py                 # Testing script
└── agent_evaluation.py           # Evaluation script
```

---

## ⚙️ Configuration

Key settings in `settings/training_settings.yaml`:

```yaml
# Simulation Parameters
tls_id: "7628244053"              # Traffic light ID
sumo_config: "sumo_ingolstadt/simulation/24h_bicycle_sim.sumocfg"
use_gui: false                     # Show SUMO GUI during training
delta_time: 5                      # Simulation step (seconds)
max_green: 60                      # Maximum green duration
min_green: 10                      # Minimum green duration
reward_type: "queue"               # Reward function

# Training Parameters
total_episodes: 200                # Number of training episodes
batch_size: 128                    # Mini-batch size
lr: 5e-5                          # Learning rate
gamma: 0.95                        # Discount factor
eps_start: 0.95                    # Initial exploration rate
eps_end: 0.1                       # Final exploration rate
```

---

## 🔧 Troubleshooting

**SUMO not found:**

```bash
# Verify SUMO installation
sumo --version

# Set SUMO_HOME environment variable
export SUMO_HOME="/usr/share/sumo"  # Adjust path as needed
```

**Out of memory during training:**

- Reduce `batch_size` to 64 in `training_settings.yaml`
- Reduce `capacity` to 50000

**Slow training:**

- Set `use_gui: false` in settings
- Reduce `max_steps` to 500 for shorter episodes

---

## 📚 How It Works

### Deep Q-Network (DQN)

The agent uses a neural network to estimate Q-values (expected future rewards) for each action:

- **State**: Lane features (queue length, waiting time, vehicle count, speed) + current phase info
- **Actions**: Continue current phase OR switch to next phase
- **Reward**: Reduction in queue length and waiting time
- **Learning**: The agent learns by trial and error, gradually improving its policy

### Training Process

1. **Exploration**: Agent randomly explores actions initially (ε-greedy policy)
2. **Experience**: Stores experiences (state, action, reward, next_state) in replay memory
3. **Learning**: Samples random mini-batches and updates the network
4. **Improvement**: Gradually exploits learned knowledge while reducing exploration

---

## 🎓 References

- **SUMO**: [Eclipse SUMO Documentation](https://sumo.dlr.de/docs/)
- **Ingolstadt Network**: [TUM-VT/sumo_ingolstadt
  ](https://github.com/TUM-VT/sumo_ingolstadt)

---

## 📝 License

This project is open source and available under the MIT License.

---

**Ready to reduce traffic congestion with AI? Start training! 🚦🤖**