"""
DQN Traffic Light Control - Deployment and Evaluation Script

This script:
1. Loads a trained DQN model
2. Runs the model on SUMO simulation (adaptive control)
3. Runs a baseline with static timing (fixed control)
4. Compares performance metrics
5. Generates comparison plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from sumo_simulation_env import SUMOSimulation
from dqn_model import DQN
import yaml


class DQNDeployment:
    def __init__(self, model_path: str, settings_file: str, output_dir: str):
        """
        Initialize deployment environment

        Args:
            model_path: Path to trained model weights (.pt file)
            settings_file: Path to settings YAML file
            output_dir: Directory to save results
        """
        self.model_path = Path(model_path)
        self.settings_file = Path(settings_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load settings
        with open(self.settings_file, "r") as f:
            self.settings = yaml.safe_load(f)

        # Device configuration
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")
        print(f"Model path: {self.model_path}")
        print(f"Output directory: {self.output_dir}")

    def load_model(self, n_observations, n_actions):
        """Load trained DQN model"""
        model = DQN(n_observations, n_actions).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()

        print(f"Model loaded successfully!")
        print(f"Observation space: {n_observations}")
        print(f"Action space: {n_actions}")

        return model

    def run_adaptive_control(self, num_episodes=5):
        """
        Run DQN adaptive control on SUMO simulation

        Args:
            num_episodes: Number of episodes to run

        Returns:
            Dictionary with results and metrics
        """
        print("\n" + "=" * 70)
        print("RUNNING ADAPTIVE CONTROL (DQN)")
        print("=" * 70)

        # Initialize environment
        env = SUMOSimulation(
            tls_id=self.settings["tls_id"],
            sumo_config=self.settings["sumo_config"],
            use_gui=self.settings.get("use_gui", False),
            delta_time=self.settings["delta_time"],
            max_green=self.settings["max_green"],
            min_green=self.settings["min_green"],
            yellow_time=self.settings["yellow_time"],
            reward_type=self.settings["reward_type"],
            max_steps=self.settings["max_steps"],
            end_on_no_vehicles=self.settings["end_on_no_vehicles"],
        )

        # Reset environment first to get dimensions
        initial_state = env.reset()
        n_observations = len(initial_state)
        n_actions = env.action_space_n

        # Load trained model with correct dimensions
        model = self.load_model(n_observations, n_actions)

        # Storage for results
        all_episodes_data = []

        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")

            # Reset for each episode (use initial_state for first episode)
            if episode == 0:
                state = initial_state
            else:
                state = env.reset(force_restart=False)

            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            done = False
            step = 0

            # Episode-specific metrics
            episode_data = {
                "timesteps": [],
                "queue_lengths": [],
                "waiting_times": [],
                "vehicles": [],
                "speeds": [],
                "rewards": [],
                "phases": [],
                "switches": [],
            }

            while not done:
                # Select action using trained model (greedy policy)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = q_values.max(1).indices.item()

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Record metrics
                episode_data["timesteps"].append(info["time"])
                episode_data["queue_lengths"].append(info["current_queue_length"])
                episode_data["waiting_times"].append(info["current_waiting_time"])
                episode_data["vehicles"].append(info["current_vehicles"])
                episode_data["speeds"].append(info["avg_speed"])
                episode_data["rewards"].append(reward)
                episode_data["phases"].append(info["current_phase"])
                episode_data["switches"].append(info["total_phase_switches"])

                # Update state
                if not done:
                    state_tensor = torch.tensor(
                        next_state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                step += 1

                # Progress logging
                if step % 100 == 0:
                    print(
                        f"  Step {step}: Queue={info['current_queue_length']:.0f}, "
                        f"Wait={info['current_waiting_time']:.1f}s, "
                        f"Vehicles={info['current_vehicles']}, "
                        f"Switches={info['total_phase_switches']}"
                    )

            # Episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"  Total Steps: {step}")
            print(f"  Avg Queue Length: {np.mean(episode_data['queue_lengths']):.2f}")
            print(f"  Avg Waiting Time: {np.mean(episode_data['waiting_times']):.2f}s")
            print(f"  Total Switches: {episode_data['switches'][-1]}")
            print(f"  Cumulative Reward: {sum(episode_data['rewards']):.2f}")

            all_episodes_data.append(episode_data)

        env.close()

        # Aggregate results across episodes
        results = self._aggregate_results(all_episodes_data, "Adaptive")

        print("\n" + "=" * 70)
        print("ADAPTIVE CONTROL COMPLETED")
        print("=" * 70)

        return results

    def run_static_control(self, num_episodes=5, green_duration=30):
        """
        Run static control (fixed timing) on SUMO simulation

        Args:
            num_episodes: Number of episodes to run
            green_duration: Fixed green phase duration in seconds

        Returns:
            Dictionary with results and metrics
        """
        print("\n" + "=" * 70)
        print(f"RUNNING STATIC CONTROL (Fixed {green_duration}s green)")
        print("=" * 70)

        # Initialize environment
        env = SUMOSimulation(
            tls_id=self.settings["tls_id"],
            sumo_config=self.settings["sumo_config"],
            use_gui=self.settings.get("use_gui", False),
            delta_time=self.settings["delta_time"],
            max_green=green_duration,
            min_green=green_duration,  # Force fixed timing
            yellow_time=self.settings["yellow_time"],
            reward_type=self.settings["reward_type"],
            max_steps=self.settings["max_steps"],
            end_on_no_vehicles=self.settings["end_on_no_vehicles"],
        )

        # Initialize environment to get dimensions
        env.reset()

        # Storage for results
        all_episodes_data = []

        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")

            state = env.reset(force_restart=(episode == 0))
            done = False
            step = 0
            time_since_switch = 0

            # Episode-specific metrics
            episode_data = {
                "timesteps": [],
                "queue_lengths": [],
                "waiting_times": [],
                "vehicles": [],
                "speeds": [],
                "rewards": [],
                "phases": [],
                "switches": [],
            }

            while not done:
                # Static control: switch every green_duration seconds
                if time_since_switch >= green_duration:
                    action = 1  # Switch
                    time_since_switch = 0
                else:
                    action = 0  # Continue

                # Execute action
                next_state, reward, done, info = env.step(action)

                time_since_switch += self.settings["delta_time"]

                # Record metrics
                episode_data["timesteps"].append(info["time"])
                episode_data["queue_lengths"].append(info["current_queue_length"])
                episode_data["waiting_times"].append(info["current_waiting_time"])
                episode_data["vehicles"].append(info["current_vehicles"])
                episode_data["speeds"].append(info["avg_speed"])
                episode_data["rewards"].append(reward)
                episode_data["phases"].append(info["current_phase"])
                episode_data["switches"].append(info["total_phase_switches"])

                step += 1

                # Progress logging
                if step % 100 == 0:
                    print(
                        f"  Step {step}: Queue={info['current_queue_length']:.0f}, "
                        f"Wait={info['current_waiting_time']:.1f}s, "
                        f"Vehicles={info['current_vehicles']}, "
                        f"Switches={info['total_phase_switches']}"
                    )

            # Episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"  Total Steps: {step}")
            print(f"  Avg Queue Length: {np.mean(episode_data['queue_lengths']):.2f}")
            print(f"  Avg Waiting Time: {np.mean(episode_data['waiting_times']):.2f}s")
            print(f"  Total Switches: {episode_data['switches'][-1]}")

            all_episodes_data.append(episode_data)

        env.close()

        # Aggregate results across episodes
        results = self._aggregate_results(all_episodes_data, "Static")

        print("\n" + "=" * 70)
        print("STATIC CONTROL COMPLETED")
        print("=" * 70)

        return results

    def _aggregate_results(self, all_episodes_data, control_type):
        """Aggregate results across multiple episodes"""
        # Find minimum length to align all episodes
        min_length = min(len(ep["timesteps"]) for ep in all_episodes_data)

        # Truncate all episodes to same length
        timesteps = all_episodes_data[0]["timesteps"][:min_length]

        # Average metrics across episodes
        queue_lengths = np.mean(
            [ep["queue_lengths"][:min_length] for ep in all_episodes_data], axis=0
        )
        waiting_times = np.mean(
            [ep["waiting_times"][:min_length] for ep in all_episodes_data], axis=0
        )
        vehicles = np.mean(
            [ep["vehicles"][:min_length] for ep in all_episodes_data], axis=0
        )
        speeds = np.mean(
            [ep["speeds"][:min_length] for ep in all_episodes_data], axis=0
        )

        # Calculate summary statistics
        summary = {
            "control_type": control_type,
            "num_episodes": len(all_episodes_data),
            "avg_queue_length": float(np.mean(queue_lengths)),
            "avg_waiting_time": float(np.mean(waiting_times)),
            "avg_vehicles": float(np.mean(vehicles)),
            "avg_speed": float(np.mean(speeds)),
            "avg_switches": float(
                np.mean([ep["switches"][-1] for ep in all_episodes_data])
            ),
            "total_steps": len(timesteps),
        }

        results = {
            "summary": summary,
            "timesteps": timesteps,
            "queue_lengths": queue_lengths.tolist(),
            "waiting_times": waiting_times.tolist(),
            "vehicles": vehicles.tolist(),
            "speeds": speeds.tolist(),
            "raw_episodes": all_episodes_data,
        }

        return results

    def plot_comparison(self, adaptive_results, static_results):
        """
        Generate comparison plots between adaptive and static control

        Args:
            adaptive_results: Results from adaptive control
            static_results: Results from static control
        """
        print("\n" + "=" * 70)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 70)

        # 1. Queue Length Comparison
        self._plot_single_metric(
            adaptive_results,
            static_results,
            metric="queue_lengths",
            ylabel="Queue Length (vehicles)",
            title="Queue Length Comparison: Adaptive vs Static Control",
            filename="queue_length_comparison.png",
        )

        # 2. Waiting Time Comparison
        self._plot_single_metric(
            adaptive_results,
            static_results,
            metric="waiting_times",
            ylabel="Waiting Time (seconds)",
            title="Waiting Time Comparison: Adaptive vs Static Control",
            filename="waiting_time_comparison.png",
        )

        # 3. Average Speed Comparison
        self._plot_single_metric(
            adaptive_results,
            static_results,
            metric="speeds",
            ylabel="Average Speed (m/s)",
            title="Average Speed Comparison: Adaptive vs Static Control",
            filename="speed_comparison.png",
        )

        # 4. Number of Vehicles Comparison
        self._plot_single_metric(
            adaptive_results,
            static_results,
            metric="vehicles",
            ylabel="Number of Vehicles",
            title="Traffic Volume Comparison: Adaptive vs Static Control",
            filename="vehicles_comparison.png",
        )

        # 5. Summary Comparison (Bar Chart)
        self._plot_summary_comparison(adaptive_results, static_results)

        print("All plots saved successfully!")

    def _plot_single_metric(
        self, adaptive_results, static_results, metric, ylabel, title, filename
    ):
        """Plot a single metric comparison"""
        plt.figure(figsize=(14, 6))

        adaptive_time = np.array(adaptive_results["timesteps"])
        static_time = np.array(static_results["timesteps"])
        adaptive_data = np.array(adaptive_results[metric])
        static_data = np.array(static_results[metric])

        plt.plot(adaptive_time, adaptive_data, alpha=0.3, color="blue", linewidth=0.5)
        plt.plot(static_time, static_data, alpha=0.3, color="red", linewidth=0.5)

        window = 50
        if len(adaptive_data) >= window:
            adaptive_smooth = np.convolve(
                adaptive_data, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                adaptive_time[window - 1 :],
                adaptive_smooth,
                label="Adaptive (DQN)",
                color="blue",
                linewidth=2,
            )
        else:
            plt.plot(
                adaptive_time,
                adaptive_data,
                label="Adaptive (DQN)",
                color="blue",
                linewidth=2,
            )

        if len(static_data) >= window:
            static_smooth = np.convolve(
                static_data, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                static_time[window - 1 :],
                static_smooth,
                label="Static Control",
                color="red",
                linewidth=2,
            )
        else:
            plt.plot(
                static_time,
                static_data,
                label="Static Control",
                color="red",
                linewidth=2,
            )

        plt.xlabel("Simulation Time (seconds)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filename}")

    def _plot_summary_comparison(self, adaptive_results, static_results):
        """Create bar chart comparing summary statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Performance Comparison: Adaptive vs Static Control",
            fontsize=16,
            fontweight="bold",
        )

        adaptive_summary = adaptive_results["summary"]
        static_summary = static_results["summary"]

        metrics = [
            ("avg_queue_length", "Average Queue Length\n(vehicles)", axes[0, 0]),
            ("avg_waiting_time", "Average Waiting Time\n(seconds)", axes[0, 1]),
            ("avg_speed", "Average Speed\n(m/s)", axes[1, 0]),
            ("avg_switches", "Total Phase Switches", axes[1, 1]),
        ]

        for metric_key, metric_label, ax in metrics:
            adaptive_val = adaptive_summary[metric_key]
            static_val = static_summary[metric_key]

            bars = ax.bar(
                ["Adaptive\n(DQN)", "Static\nControl"],
                [adaptive_val, static_val],
                color=["#2ecc71", "#e74c3c"],
                edgecolor="black",
                linewidth=1.5,
            )

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            ax.set_ylabel(metric_label, fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")

            # Calculate improvement
            if metric_key in ["avg_queue_length", "avg_waiting_time"]:
                # Lower is better
                improvement = ((static_val - adaptive_val) / static_val) * 100
                color = "green" if improvement > 0 else "red"
            else:
                # Higher is better for speed
                improvement = ((adaptive_val - static_val) / static_val) * 100
                color = "green" if improvement > 0 else "red"

            if metric_key != "avg_switches":
                ax.set_title(
                    f"{improvement:+.1f}% improvement",
                    color=color,
                    fontsize=10,
                    fontweight="bold",
                )

        plt.tight_layout()

        save_path = self.output_dir / "summary_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✓ Saved: summary_comparison.png")

    def save_results(self, adaptive_results, static_results):
        """Save results to JSON file"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "adaptive": adaptive_results["summary"],
            "static": static_results["summary"],
            "settings": self.settings,
        }

        # Calculate improvements
        results["improvements"] = {
            "queue_length_reduction": float(
                (
                    (
                        static_results["summary"]["avg_queue_length"]
                        - adaptive_results["summary"]["avg_queue_length"]
                    )
                    / static_results["summary"]["avg_queue_length"]
                )
                * 100
            ),
            "waiting_time_reduction": float(
                (
                    (
                        static_results["summary"]["avg_waiting_time"]
                        - adaptive_results["summary"]["avg_waiting_time"]
                    )
                    / static_results["summary"]["avg_waiting_time"]
                )
                * 100
            ),
            "speed_improvement": float(
                (
                    (
                        adaptive_results["summary"]["avg_speed"]
                        - static_results["summary"]["avg_speed"]
                    )
                    / static_results["summary"]["avg_speed"]
                )
                * 100
            ),
        }

        save_path = self.output_dir / "evaluation_results.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {save_path}")

        return results

    def print_summary(self, results):
        """Print comprehensive summary of evaluation"""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        print("\nADAPTIVE CONTROL (DQN):")
        print(
            f"  Average Queue Length:  {results['adaptive']['avg_queue_length']:.2f} vehicles"
        )
        print(
            f"  Average Waiting Time:  {results['adaptive']['avg_waiting_time']:.2f} seconds"
        )
        print(f"  Average Speed:         {results['adaptive']['avg_speed']:.2f} m/s")
        print(f"  Total Phase Switches:  {results['adaptive']['avg_switches']:.0f}")

        print("\nSTATIC CONTROL:")
        print(
            f"  Average Queue Length:  {results['static']['avg_queue_length']:.2f} vehicles"
        )
        print(
            f"  Average Waiting Time:  {results['static']['avg_waiting_time']:.2f} seconds"
        )
        print(f"  Average Speed:         {results['static']['avg_speed']:.2f} m/s")
        print(f"  Total Phase Switches:  {results['static']['avg_switches']:.0f}")

        print("\nIMPROVEMENTS:")
        print(
            f"  Queue Length Reduction:  {results['improvements']['queue_length_reduction']:+.1f}%"
        )
        print(
            f"  Waiting Time Reduction:  {results['improvements']['waiting_time_reduction']:+.1f}%"
        )
        print(
            f"  Speed Improvement:       {results['improvements']['speed_improvement']:+.1f}%"
        )

        print("\n" + "=" * 70)


def main():
    """Main deployment script"""
    print("\n" + "=" * 70)
    print("DQN TRAFFIC LIGHT CONTROL - DEPLOYMENT & EVALUATION")
    print("=" * 70 + "\n")

    # Configuration
    MODEL_PATH = "training_outputs/dqn_sumo_policy_final_400.pt"
    SETTINGS_FILE = "settings/training_settings.yaml"
    OUTPUT_DIR = "testing_outputs"
    NUM_EPISODES = 5  # Number of episodes to average
    STATIC_GREEN_DURATION = 30  # Fixed green time for static control (seconds)

    # Initialize deployment
    deployment = DQNDeployment(
        model_path=MODEL_PATH, settings_file=SETTINGS_FILE, output_dir=OUTPUT_DIR
    )

    try:
        # Run adaptive control
        adaptive_results = deployment.run_adaptive_control(num_episodes=NUM_EPISODES)

        # Run static control
        static_results = deployment.run_static_control(
            num_episodes=NUM_EPISODES, green_duration=STATIC_GREEN_DURATION
        )

        # Generate comparison plots
        deployment.plot_comparison(adaptive_results, static_results)

        # Save results
        results = deployment.save_results(adaptive_results, static_results)

        # Print summary
        deployment.print_summary(results)

        print("\n✓ Deployment completed successfully!")
        print(f"✓ All results saved to: {OUTPUT_DIR}/")

    except Exception as e:
        print(f"\n✗ Error during deployment: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
