from sumo_simulation_env import SUMOSimulation
from model import DQN, ReplayMemory, Transition
import random
import yaml
from pathlib import Path
import math
from itertools import count
import matplotlib
import matplotlib.pyplot as plt

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
    else "mps" if torch.mps.is_available() else "cpu"
)


class DQNTraining:
    def __init__(self, settings_file: Path, output_dir: Path):
        self.settings_file = settings_file
        self.output_dir = output_dir
        self.settings = self.load_settings(self.settings_file)

        self.sumo_config = self.settings["sumo_config"]
        self.tls_id = self.settings["tls_id"]
        self.use_gui = self.settings["use_gui"]
        self.delta_time = self.settings["delta_time"]
        self.max_green = self.settings["max_green"]
        self.min_green = self.settings["min_green"]
        self.reward_type = self.settings["reward_type"]
        self.max_steps = self.settings["max_steps"]
        self.end_on_no_vehicle = self.settings["end_on_no_vehicles"]
        self.capacity = self.settings["capacity"]
        self.num_episodes = self.settings["total_episodes"]

        self.batch_size = self.settings["batch_size"]
        self.gamma = self.settings["gamma"]
        self.eps_start = self.settings["eps_start"]
        self.eps_end = self.settings["eps_end"]
        self.eps_decay = self.settings["eps_decay"]
        self.tau = self.settings["tau"]
        self.lr = self.settings["lr"]
        self.steps_done = 0

        self.memory = ReplayMemory(self.capacity)
        self.env = SUMOSimulation(
            self.tls_id,
            self.sumo_config,
            self.use_gui,
            self.delta_time,
            self.max_green,
            self.min_green,
            self.reward_type,
            self.max_steps,
            self.end_on_no_vehicle,
        )

        self.state = self.env.reset()
        self.n_observations = len(self.state)
        self.n_actions = self.env.action_space_n

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=float(self.lr), amsgrad=True
        )

        self.episode_returns = []
        self.episode_durations = []

    def load_settings(self, file_path):
        with open(file_path, "r") as file:
            settings_dict = yaml.safe_load(file)
        return settings_dict

    def start_training(self):
        for i_episode in range(self.num_episodes):
            self.steps_done = 0
            print(f"Episode {i_episode} started")
            # reset SUMO Environment
            state_np = self.env.reset(force_restart=True)
            state = torch.tensor(
                state_np, dtype=torch.float32, device=device
            ).unsqueeze(0)

            episode_return = 0.0

            for t in count():
                action = self.select_action(state)

                # step environment
                next_state_np, reward_val, done, info = self.env.step(action.item())

                reward = torch.tensor([reward_val], device=device)
                episode_return += reward_val

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        next_state_np, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                if self.steps_done % 100 == 0:
                    print(
                        f"Steps finished {self.steps_done} | "
                        f"Steps: {t + 1} | Return: {episode_return:.2f} | "
                        f"Switches: {info['total_phase_switches']} | "
                        f"Cumulative waiting: {info['cumulative_waiting_time']:.2f}"
                    )

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * self.tau + target_net_state_dict[key] * (1.0 - self.tau)

                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_returns.append(episode_return)
                    self.episode_durations.append(t + 1)
                    print(
                        f"Episode {i_episode + 1}/{self.num_episodes} | "
                        f"Steps: {t + 1} | Return: {episode_return:.2f} | "
                        f"Switches: {info['total_phase_switches']} | "
                        f"Cumulative waiting: {info['cumulative_waiting_time']:.2f}"
                    )
                    break

        torch.save(self.policy_net.state_dict(), "dqn_sumo_policy.pt")

        self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()

        print("Training complete.")
        self.env.close()

    def select_action(self, state):
        sample = random.random()

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]], device=device, dtype=torch.long
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Computer Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


if __name__ == "__main__":
    settings_file = Path("settings/training_settings.yaml")
    output_dir = Path("training_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_session = DQNTraining(settings_file, output_dir)
    training_session.start_training()
