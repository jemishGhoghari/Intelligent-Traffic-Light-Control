from sumo_simulation_env import SUMOSimulation
from model import DQN, ReplayMemory, Transition
import random
import yaml
from pathlib import Path
import math
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

import torch
from torch import nn
import torch.optim

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# check if CUDA available
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class DQNTraining:
    def __init__(self, settings_file: Path, output_dir: Path):
        self.settings_file = settings_file
        self.output_dir = output_dir
        self.settings = self.load_settings(self.settings_file)

        # SUMO environment parameters
        self.sumo_config = self.settings["sumo_config"]
        self.tls_id = self.settings["tls_id"]
        self.use_gui = self.settings["use_gui"]
        self.delta_time = self.settings["delta_time"]
        self.max_green = self.settings["max_green"]
        self.min_green = self.settings["min_green"]
        self.yellow_time = self.settings["yellow_time"]
        self.reward_type = self.settings["reward_type"]
        self.max_steps = self.settings["max_steps"]
        self.end_on_no_vehicle = self.settings["end_on_no_vehicles"]

        # DQN parameters
        self.capacity = self.settings["capacity"]
        self.num_episodes = self.settings["total_episodes"]
        self.batch_size = self.settings["batch_size"]
        self.gamma = self.settings["gamma"]
        self.eps_start = self.settings["eps_start"]
        self.eps_end = self.settings["eps_end"]
        self.eps_decay = self.settings["eps_decay"]
        self.tau = self.settings["tau"]
        self.lr = self.settings["lr"]

        # NEW: Learning warmup - don't train until we have enough samples
        self.learning_starts = self.settings.get("learning_starts", 1000)

        # NEW: Update frequency - update target network less often
        self.target_update_frequency = self.settings.get(
            "target_update_frequency", 1000
        )

        # NEW: Reward clipping for stability
        self.reward_clip = self.settings.get("reward_clip", 10.0)

        # Training state
        self.steps_done = 0
        self.updates_done = 0

        # Tracking metrics
        self.episode_returns = []
        self.episode_durations = []
        self.episode_waiting_times = []
        self.episode_switches = []
        self.training_losses = []
        self.step_losses = []  # Track every optimization step

        # NEW: Track Q-values for monitoring
        self.avg_q_values = []
        self.max_q_values = []

        # Initialize replay memory
        self.memory = ReplayMemory(self.capacity)

        # Initialize SUMO environment
        print("Initializing SUMO environment...")
        self.env = SUMOSimulation(
            tls_id=self.tls_id,
            sumo_config=self.sumo_config,
            use_gui=self.use_gui,
            delta_time=self.delta_time,
            max_green=self.max_green,
            min_green=self.min_green,
            yellow_time=self.yellow_time,
            reward_type=self.reward_type,
            max_steps=self.max_steps,
            end_on_no_vehicles=self.end_on_no_vehicle,
        )

        # Get observation and action space dimensions
        self.state = self.env.reset()
        self.n_observations = len(self.state)
        self.n_actions = self.env.action_space_n

        print(f"Observation space: {self.n_observations}")
        print(f"Action space: {self.n_actions}")

        # Initialize policy and target networks
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=float(self.lr), amsgrad=True
        )

        print(f"Using device: {device}")
        print(f"Learning starts after: {self.learning_starts} steps")
        print(f"Target network update frequency: {self.target_update_frequency} steps")
        print(f"Model architecture:\n{self.policy_net}")

    def load_settings(self, file_path):
        """Load training settings from YAML file"""
        with open(file_path, "r") as file:
            settings_dict = yaml.safe_load(file)
        return settings_dict

    def start_training(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("STARTING DQN TRAINING")
        print("=" * 70)
        print(f"Total episodes: {self.num_episodes}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Gamma: {self.gamma}")
        print(f"Epsilon: {self.eps_start} -> {self.eps_end} (decay: {self.eps_decay})")
        print(f"Reward type: {self.reward_type}")
        print(f"Reward clipping: ±{self.reward_clip}")
        print("=" * 70 + "\n")

        training_start_time = datetime.now()

        for i_episode in range(self.num_episodes):
            episode_start_time = datetime.now()

            print(f"\n{'='*70}")
            print(f"EPISODE {i_episode + 1}/{self.num_episodes}")
            print(f"Current epsilon: {self._get_epsilon():.4f}")
            print(f"Memory size: {len(self.memory)}/{self.capacity}")
            print(f"Total steps: {self.steps_done} | Updates: {self.updates_done}")
            print(f"{'='*70}")

            # Reset environment
            state_np = self.env.reset(force_restart=True)
            state = torch.tensor(
                state_np, dtype=torch.float32, device=device
            ).unsqueeze(0)

            episode_return = 0.0
            episode_losses = []
            episode_q_values = []

            # Episode loop
            for t in count():
                # Select and perform action
                action, q_value = self.select_action(state, return_q=True)
                if q_value is not None:
                    episode_q_values.append(q_value)

                next_state_np, reward_val, done, info = self.env.step(action.item())

                # NEW: Clip rewards for stability
                reward_val = np.clip(reward_val, -self.reward_clip, self.reward_clip)

                reward = torch.tensor([reward_val], device=device)
                episode_return += reward_val

                # Process next state
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        next_state_np, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                # Store transition in replay memory
                self.memory.push(state, action, next_state, reward)

                # Move to next state
                state = next_state

                # Perform optimization step (only after warmup)
                if self.steps_done >= self.learning_starts:
                    loss = self.optimize_model()
                    if loss is not None:
                        episode_losses.append(loss)
                        self.step_losses.append(loss)

                    # NEW: Update target network periodically instead of every step
                    if self.updates_done % self.target_update_frequency == 0:
                        self._hard_update_target_network()
                        print(f"    → Target network updated at step {self.steps_done}")

                # Logging
                if t > 0 and t % 50 == 0:
                    avg_loss = np.mean(episode_losses[-50:]) if episode_losses else 0
                    avg_q = np.mean(episode_q_values[-50:]) if episode_q_values else 0
                    print(
                        f"  Step {t:4d} | Return: {episode_return:8.2f} | "
                        f"Switches: {info['total_phase_switches']:3d} | "
                        f"Waiting: {info['cumulative_waiting_time']:7.1f}s | "
                        f"Loss: {avg_loss:.4f} | Avg Q: {avg_q:.2f}"
                    )

                if done:
                    # Record episode metrics
                    self.episode_returns.append(episode_return)
                    self.episode_durations.append(t + 1)
                    self.episode_waiting_times.append(info["cumulative_waiting_time"])
                    self.episode_switches.append(info["total_phase_switches"])

                    if episode_losses:
                        self.training_losses.append(np.mean(episode_losses))

                    if episode_q_values:
                        self.avg_q_values.append(np.mean(episode_q_values))
                        self.max_q_values.append(np.max(episode_q_values))

                    episode_duration = datetime.now() - episode_start_time

                    print(f"\n{'='*70}")
                    print(f"EPISODE {i_episode + 1} COMPLETED")
                    print(f"Duration: {episode_duration.total_seconds():.1f}s")
                    print(f"Steps: {t + 1}")
                    print(f"Total Return: {episode_return:.2f}")
                    print(f"Total Waiting Time: {info['cumulative_waiting_time']:.2f}s")
                    print(f"Phase Switches: {info['total_phase_switches']}")
                    if episode_losses:
                        print(f"Avg Loss: {np.mean(episode_losses):.4f}")
                    if episode_q_values:
                        print(f"Avg Q-value: {np.mean(episode_q_values):.2f}")
                        print(f"Max Q-value: {np.max(episode_q_values):.2f}")
                    print(f"{'='*70}")

                    # Plot progress every 10 episodes
                    if (i_episode + 1) % 10 == 0:
                        self._save_checkpoint(i_episode + 1)
                        self.plot_all_metrics()

                    break

        # Training complete
        training_duration = datetime.now() - training_start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Total training time: {training_duration}")
        print(f"Total episodes: {self.num_episodes}")
        print(f"Total steps: {self.steps_done}")
        print(f"Total updates: {self.updates_done}")
        print(f"Final epsilon: {self._get_epsilon():.4f}")
        print("=" * 70 + "\n")

        # Save final model and results
        self._save_final_model()

        # Plot final results
        self.plot_all_metrics()

        plt.ioff()
        plt.show()

        print("\nClosing environment...")
        self.env.close()
        print("Training session finished successfully!")

    def _get_epsilon(self):
        """Calculate current epsilon value for epsilon-greedy policy"""
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )

    def select_action(self, state, return_q=False):
        """Select action using epsilon-greedy policy"""
        sample = random.random()
        eps_threshold = self._get_epsilon()
        self.steps_done += 1

        q_value = None

        if sample > eps_threshold:
            # Exploit: choose best action from policy network
            with torch.no_grad():
                q_values = self.policy_net(state)
                q_value = q_values.max(1).values.item()
                action = q_values.max(1).indices.view(1, 1)
        else:
            # Explore: choose random action
            action = torch.tensor(
                [[random.randrange(self.n_actions)]], device=device, dtype=torch.long
            )

        if return_q:
            return action, q_value
        return action

    def optimize_model(self):
        """Perform one step of optimization on the policy network"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from replay memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute mask for non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) using target network
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # NEW: Gradient clipping with norm instead of value
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()
        self.updates_done += 1

        return loss.item()

    def _hard_update_target_network(self):
        """Hard update of target network (copy all weights)"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _save_checkpoint(self, episode):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_ep{episode}.pt"
        torch.save(
            {
                "episode": episode,
                "model_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "updates_done": self.updates_done,
                "episode_returns": self.episode_returns,
                "episode_durations": self.episode_durations,
                "training_losses": self.training_losses,
            },
            checkpoint_path,
        )
        print(f"    → Checkpoint saved: {checkpoint_path}")

    def _save_final_model(self):
        """Save final trained model"""
        model_path = self.output_dir / "dqn_sumo_policy_final.pt"
        torch.save(self.policy_net.state_dict(), model_path)
        print(f"Final model saved: {model_path}")

        # Also save complete checkpoint
        final_checkpoint = self.output_dir / "final_checkpoint.pt"
        torch.save(
            {
                "episode": self.num_episodes,
                "model_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "updates_done": self.updates_done,
                "episode_returns": self.episode_returns,
                "episode_durations": self.episode_durations,
                "settings": self.settings,
            },
            final_checkpoint,
        )
        print(f"Final checkpoint saved: {final_checkpoint}")

    def plot_all_metrics(self):
        """Save all training metrics as a single figure (no GUI)."""

        # Optional but useful if you ever run in an interactive environment
        plt.ioff()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("DQN Training Metrics", fontsize=16, fontweight="bold")

        # 1. Episode Returns
        ax = axes[0, 0]
        if self.episode_returns:
            returns_t = torch.tensor(self.episode_returns, dtype=torch.float)
            ax.plot(returns_t.numpy(), alpha=0.5, label="Return", color="green")
            if len(returns_t) >= 10:
                window = min(50, len(returns_t))
                means = returns_t.unfold(0, window, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(window - 1), means))
                ax.plot(
                    means.numpy(),
                    label=f"{window}-Ep Avg",
                    linewidth=2,
                    color="darkgreen",
                )
            ax.set_title("Episode Returns")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Return")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Episode Durations
        ax = axes[0, 1]
        if self.episode_durations:
            durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
            ax.plot(durations_t.numpy(), alpha=0.5, label="Duration", color="blue")
            if len(durations_t) >= 10:
                window = min(50, len(durations_t))
                means = durations_t.unfold(0, window, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(window - 1), means))
                ax.plot(
                    means.numpy(),
                    label=f"{window}-Ep Avg",
                    linewidth=2,
                    color="darkblue",
                )
            ax.set_title("Episode Durations")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Steps")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Training Losses
        ax = axes[0, 2]
        if self.training_losses:
            ax.plot(self.training_losses, alpha=0.7, label="Avg Loss", color="orange")
            if len(self.training_losses) >= 10:
                window = min(50, len(self.training_losses))
                losses_array = np.array(self.training_losses)
                moving_avg = np.convolve(
                    losses_array, np.ones(window) / window, mode="valid"
                )
                ax.plot(
                    range(window - 1, len(self.training_losses)),
                    moving_avg,
                    label=f"{window}-Ep Avg",
                    linewidth=2,
                    color="red",
                )
            ax.set_title("Training Loss per Episode")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Waiting Times
        ax = axes[1, 0]
        if self.episode_waiting_times:
            ax.plot(
                self.episode_waiting_times,
                alpha=0.5,
                label="Waiting Time",
                color="purple",
            )
            if len(self.episode_waiting_times) >= 10:
                window = min(50, len(self.episode_waiting_times))
                waiting_array = np.array(self.episode_waiting_times)
                moving_avg = np.convolve(
                    waiting_array, np.ones(window) / window, mode="valid"
                )
                ax.plot(
                    range(window - 1, len(self.episode_waiting_times)),
                    moving_avg,
                    label=f"{window}-Ep Avg",
                    linewidth=2,
                    color="darkviolet",
                )
            ax.set_title("Cumulative Waiting Time")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Time (s)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 5. Q-values
        ax = axes[1, 1]
        if self.avg_q_values:
            ax.plot(self.avg_q_values, alpha=0.7, label="Avg Q", color="cyan")
            ax.plot(self.max_q_values, alpha=0.7, label="Max Q", color="teal")
            ax.set_title("Q-values per Episode")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Q-value")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 6. Step-wise Loss (last 5000 steps)
        ax = axes[1, 2]
        if len(self.step_losses) > 0:
            recent_losses = (
                self.step_losses[-5000:]
                if len(self.step_losses) > 5000
                else self.step_losses
            )
            ax.plot(recent_losses, alpha=0.3, color="orange")
            if len(recent_losses) >= 100:
                window = 100
                moving_avg = np.convolve(
                    np.array(recent_losses), np.ones(window) / window, mode="valid"
                )
                ax.plot(
                    range(window - 1, len(recent_losses)),
                    moving_avg,
                    label=f"{window}-Step Avg",
                    linewidth=2,
                    color="red",
                )
                ax.legend()
            ax.set_title("Recent Step-wise Loss")
            ax.set_xlabel("Update Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

        # Layout + save
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "training_metrics.png", dpi=300, bbox_inches="tight"
        )

        # Important: free resources (prevents memory leak during long training)
        plt.close(fig)


def main():
    """Main entry point for training"""
    settings_file = Path("settings/training_settings.yaml")
    output_dir = Path("training_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DQN TRAFFIC LIGHT CONTROL - TRAINING SESSION")
    print("=" * 70)
    print(f"Settings file: {settings_file}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("=" * 70 + "\n")

    try:
        training_session = DQNTraining(settings_file, output_dir)
        training_session.start_training()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current progress...")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nTraining session ended.")


if __name__ == "__main__":
    main()
