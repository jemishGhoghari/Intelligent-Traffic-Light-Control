from sumo_simulation_env import SUMOSimulation
from dqn_model import DQN
import torch
import yaml
import matplotlib.pyplot as plt

# Device configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class LiveTrafficVisualizer:
    """Real-time visualization of traffic metrics while model runs"""

    def __init__(self, max_steps=1000):
        self.max_steps = max_steps

        self.steps = []
        self.queue_lengths = []
        self.num_vehicles = []
        self.avg_speeds = []
        self.rewards = []
        self.actions = []

        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle(
            "DQN Traffic Light Control - Live Metrics", fontsize=14, fontweight="bold"
        )

        self.lines = {}

        ax = self.axes[0, 0]
        (self.lines["queue"],) = ax.plot(
            [], [], "r-", linewidth=2, label="Queue Length"
        )
        ax.set_xlim(0, max_steps)
        ax.set_ylim(0, 50)
        ax.set_xlabel("Step")
        ax.set_ylabel("Queue Length (vehicles)")
        ax.set_title("Queue Length Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = self.axes[0, 1]
        (self.lines["vehicles"],) = ax.plot([], [], "b-", linewidth=2, label="Vehicles")
        ax.set_xlim(0, max_steps)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Step")
        ax.set_ylabel("Number of Vehicles")
        ax.set_title("Number of Vehicles Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = self.axes[1, 0]
        (self.lines["speed"],) = ax.plot([], [], "g-", linewidth=2, label="Avg Speed")
        ax.set_xlim(0, max_steps)
        ax.set_ylim(0, 20)
        ax.set_xlabel("Step")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Average Speed Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = self.axes[1, 1]
        (self.lines["reward"],) = ax.plot(
            [], [], "purple", linewidth=2, label="Cumulative Reward"
        )
        ax.set_xlim(0, max_steps)
        ax.set_ylim(-200, 50)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Cumulative Reward Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.ion()
        plt.show()

    def update(self, step, queue, vehicles, speed, reward, action):
        """Update plots with new data"""
        self.steps.append(step)
        self.queue_lengths.append(queue)
        self.num_vehicles.append(vehicles)
        self.avg_speeds.append(speed)

        if len(self.rewards) == 0:
            self.rewards.append(reward)
        else:
            self.rewards.append(self.rewards[-1] + reward)

        self.actions.append(action)

        self.lines["queue"].set_data(self.steps, self.queue_lengths)
        self.lines["vehicles"].set_data(self.steps, self.num_vehicles)
        self.lines["speed"].set_data(self.steps, self.avg_speeds)
        self.lines["reward"].set_data(self.steps, self.rewards)

        if len(self.queue_lengths) > 0:
            max_queue = max(self.queue_lengths)
            if max_queue > self.axes[0, 0].get_ylim()[1] * 0.9:
                self.axes[0, 0].set_ylim(0, max_queue * 1.2)

        if len(self.num_vehicles) > 0:
            max_veh = max(self.num_vehicles)
            if max_veh > self.axes[0, 1].get_ylim()[1] * 0.9:
                self.axes[0, 1].set_ylim(0, max_veh * 1.2)

        if len(self.rewards) > 0:
            min_rew = min(self.rewards)
            max_rew = max(self.rewards)
            if min_rew < self.axes[1, 1].get_ylim()[0]:
                self.axes[1, 1].set_ylim(min_rew * 1.2, max_rew * 1.2)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def test_model(
    model_path: str,
    settings_file: str = "settings/training_settings.yaml",
    use_gui: bool = True,
    visualize: bool = True,
    max_steps: int = 1000,
):
    """
    Test a trained model on the traffic simulation.

    Args:
        model_path: Path to trained model (.pt file)
        settings_file: Path to settings YAML
        use_gui: Show SUMO GUI
        visualize: Show live matplotlib visualization
        max_steps: Maximum steps to run (None = until episode ends naturally)
    """
    print("\n" + "=" * 70)
    print("TESTING TRAINED DQN MODEL")

    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    if max_steps is not None:
        settings["max_steps"] = max_steps

    print("Initializing environment...")
    env = SUMOSimulation(
        tls_id=settings["tls_id"],
        sumo_config=settings["sumo_config"],
        use_gui=use_gui,
        delta_time=settings["delta_time"],
        max_green=settings["max_green"],
        min_green=settings["min_green"],
        yellow_time=settings.get("yellow_time", 3),
        reward_type=settings["reward_type"],
        max_steps=settings["max_steps"],
        end_on_no_vehicles=settings["end_on_no_vehicles"],
    )

    state = env.reset()
    n_observations = len(state)
    n_actions = env.action_space_n

    print(f"Observation space: {n_observations}")
    print(f"Action space: {n_actions}")

    print(f"\nLoading model from {model_path}...")
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()
    print("Model loaded successfully!")

    if visualize:
        viz = LiveTrafficVisualizer(max_steps=settings["max_steps"])

    print("\n" + "=" * 70)
    print("STARTING SIMULATION")
    print("=" * 70 + "\n")

    state = env.reset(force_restart=True)
    done = False
    step = 0
    episode_return = 0.0

    action_counts = {0: 0, 1: 0}

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.max(1).indices.item()
            max_q = q_values.max(1).values.item()

        next_state, reward, done, info = env.step(action)
        episode_return += reward
        step += 1

        action_counts[action] += 1

        if visualize:
            viz.update(
                step=step,
                queue=info["current_queue_length"],
                vehicles=info["current_vehicles"],
                speed=info["avg_speed"],
                reward=reward,
                action=action,
            )

        action_str = "CONTINUE" if action == 0 else "SWITCH"
        print(
            f"Step {step:4d} | Action: {action_str:8s} | Q: {max_q:7.2f} | "
            f"Queue: {info['current_queue_length']:4.0f} | "
            f"Vehicles: {info['current_vehicles']:4.0f} | "
            f"Speed: {info['avg_speed']:5.2f}m/s | "
            f"Reward: {reward:7.2f} | "
            f"Switches: {info['total_phase_switches']:3d}"
        )

        state = next_state

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED")
    print("=" * 70)
    print(f"Total Steps: {step}")
    print(f"Total Return: {episode_return:.2f}")
    print(f"Total Phase Switches: {info['total_phase_switches']}")
    print(f"Cumulative Waiting Time: {info['cumulative_waiting_time']:.2f}s")
    print(f"\nAction Distribution:")
    print(f"  Continue (0): {action_counts[0]} ({action_counts[0]/step*100:.1f}%)")
    print(f"  Switch (1): {action_counts[1]} ({action_counts[1]/step*100:.1f}%)")
    print("=" * 70 + "\n")

    if visualize:
        print("Close the visualization window to exit...")
        plt.ioff()
        plt.show()

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test trained DQN traffic light model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="settings/training_settings.yaml",
        help="Path to training settings YAML file",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable SUMO GUI")
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable live matplotlib visualization"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Maximum number of steps to run"
    )

    args = parser.parse_args()

    test_model(
        model_path=args.model,
        settings_file=args.settings,
        use_gui=not args.no_gui,
        visualize=not args.no_viz,
        max_steps=args.steps,
    )
