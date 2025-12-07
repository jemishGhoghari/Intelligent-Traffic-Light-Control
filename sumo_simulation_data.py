import timeit
import random
import numpy as np
import os
import sys


if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

if sys.platform == "linux":
    import libsumo as traci
elif sys.platform == "win32":
    import traci
else:
    import libsumo


# sumoBinary = "sumo-gui"  # or "sumo-gui" for graphical version
# sumoConfig = "./sumo_ingolstadt/simulation/24h_bicycle_sim.sumocfg"
# sumoCmd = [sumoBinary, "-c", sumoConfig]


class SUMOSimulation:

    def __init__(
        self,
        tls_id: str,
        sumo_config,
        use_gui: bool = True,
        delta_time: int = 5,
        max_green: int = 60,
        min_green: int = 5,
        reward_type: str = "combined",
        max_steps: int = 3600,
        end_on_no_vehicles: bool = True,
    ):
        self.delta_time = delta_time
        self.tls_id = tls_id
        self.max_green = max_green
        self.min_green = min_green
        self.reward_type = reward_type
        self.sumo_config = sumo_config
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.end_on_no_vehicles = end_on_no_vehicles

        self.current_step = 0
        self.time_since_last_phase_change = 0
        self.current_phase_duration = 0

        self.total_phase_switches = 0
        self.cumulative_waiting_time = 0
        self.cumulative_reward = 0

        self.sumo_running = False
        self._start_sumo()

        # Action space
        self.action_space_n = 2
        self.observation_space_n = None

        self.num_phases = self._get_num_phases(self.tls_id)

    def _get_lane_features(self, lane_id: int):
        """
        Get Lane features like Queue Length, Waiting time of the vehicle, Number of Vehicle on the Lane,
        Mean speed of the vehcles, and delay.

        Args:
            lane_id (int): lane id of the incoming or outgoing edge
        """

        # Get total queue length at the stopline of specific lane
        queue_length = self._get_queue_length_near_stopline(
            lane_id, distance_from_stop=40
        )

        # Calculate Vehicle Waiting times and Number of Vehicles.
        waiting_time = traci.lane.getWaitingTime(lane_id)
        num_vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
        mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)

        # Delay: Calculate the delay -> realistic travel time - expected travel time
        edge_id = traci.lane.getEdgeID(lane_id)
        t_actual = traci.edge.getTraveltime(edge_id)
        lane_length = traci.lane.getLength(lane_id)
        v_max = traci.lane.getMaxSpeed(lane_id)
        t_expected = lane_length / v_max if v_max > 0 else 0.0
        delay = max(0.0, t_actual - t_expected)  # [s]

        return np.array(
            [
                queue_length,
                waiting_time,
                num_vehicles,
                mean_speed,
                delay,
            ],
            dtype=float,
        )

    def _get_queue_length_near_stopline(self, lane_id: int, distance_from_stop: float):
        """_summary_

        Args:
            lane_id (int): lane id of incoming or outgoing link
            distance_from_stop (float): queue length within given distance from stopline
        """

        lane_length = traci.lane.getLength(lane_id)
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)

        q = 0

        for v_id in vehicle_ids:
            pos = traci.vehicle.getLanePosition(v_id)
            speed = traci.vehicle.getSpeed(v_id)

            if lane_length - pos <= distance_from_stop and speed < 0.1:
                q += 1

        return q

    def _get_num_phases(self, tls_id: str):
        current_prog = traci.trafficlight.getProgram(tls_id)

        for logic in traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id):
            if logic.programID == current_prog:
                return len(logic.phases)

        # fallback
        return len(
            traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases
        )

    def _get_state(self):
        """
        Extract State of the Intersection including Lane features and Phase features
        """

        # Extract lane features
        lanes = sorted(set(traci.trafficlight.getControlledLanes(self.tls_id)))
        lane_features = [self._get_lane_features(l) for l in lanes]
        lane_features = np.concatenate(
            lane_features
        )  # shape = 5 * n_lanes (n_lanes = 30 for Ingolstadt Saturn Arena Intersection including Bicycle lanes)

        # Extract phase features
        num_phases = self._get_num_phases(self.tls_id)

        phase_idx = traci.trafficlight.getPhase(self.tls_id)
        phase_one_hot = np.zeros(num_phases, dtype=float)
        phase_one_hot[phase_idx] = 1.0

        phase_total = traci.trafficlight.getPhaseDuration(self.tls_id)
        phase_spent = traci.trafficlight.getSpentDuration(self.tls_id)
        sim_time = traci.simulation.getTime()
        phase_remain = max(
            0.0, traci.trafficlight.getNextSwitch(self.tls_id) - sim_time
        )  # get exact time at which transition is scheduled

        # Normalize times to something [0, 1] to Help Neural Network
        max_phase_time = max(1.0, phase_total)
        phase_spent_norm = phase_spent / max_phase_time
        phase_remain_norm = phase_remain / max_phase_time

        phase_features = np.concatenate(
            [
                phase_one_hot,
                np.array([phase_spent_norm, phase_remain_norm], dtype=float),
            ]
        )

        state = np.concatenate([lane_features, phase_features])
        return state

    def _apply_action(self, action: int):
        """
        Apply the continue/switch action to the traffic light.

        Action formulation:
            - action 0: continue current phase
            - action 1: switch to next phase

        Args:
            action (int): 0 (continue) or 1 (switch)
        """
        if action == 0:
            if self.time_since_last_phase_change < self.max_green:
                pass
            else:
                self._switch_to_next_phase()
        elif action == 1:
            if self.time_since_last_phase_change >= self.min_green:
                self._switch_to_next_phase()

    def _switch_to_next_phase(self):
        """
        Switch to the next phase in the traffic light cycle
        """
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        next_phase = (current_phase + 1) % self.num_phases

        traci.trafficlight.setPhase(self.tls_id, next_phase)

        self.time_since_last_phase_change = 0
        self.total_phase_switches += 1

    def _compute_reward(self) -> float:
        """
        Calculate reward based on traffic matrics

        Returns:
            float: reward value
        """
        metrics = self._calculate_metrics()

        if self.reward_type == "waiting_time":
            reward = -metrics["total_waiting_time"]
        elif self.reward_type == "queue":
            reward = -metrics["total_queue_length"]
        elif self.reward_type == "comboned":
            reward = -metrics["total_delay"]
        elif self.reward_type == "combined":
            # weighted sum of all (We need to tune the weights)
            w_wait = 0.4
            w_queue = 0.3
            w_delay = 0.2
            w_speed = 0.1

            reward = (
                -w_wait * metrics["total_waiting_time"]
                - w_queue * metrics["total_queue_length"]
                - w_delay * metrics["total_delay"]
                + w_speed * metrics["avg_speed"] * metrics["total_vehicles"]
            )
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        self.cumulative_waiting_time += metrics["total_waiting_time"]

        return reward

    def _calculate_metrics(self) -> dict[str, float]:
        """
        Calculate traffic metrics for reward calculation

        Returns:
            Dictionary containing traffic metrics
        """
        lanes = sorted(set(traci.trafficlight.getControlledLanes(self.tls_id)))

        total_waiting_time = 0
        total_queue_length = 0
        total_delay = 0
        total_vehicles = 0
        total_speed = 0

        for lane in lanes:
            total_waiting_time += traci.lane.getWaitingTime(lane)
            total_queue_length += self._get_queue_length_near_stopline(lane, 40)
            num_veh = traci.lane.getLastStepVehicleNumber(lane)
            total_vehicles += num_veh

            if num_veh > 0:
                total_speed += traci.lane.getLastStepMeanSpeed(lane) * num_veh

            edge_id = traci.lane.getEdgeID(lane)
            t_actual = traci.edge.getTraveltime(edge_id)
            lane_length = traci.lane.getLength(lane)
            v_max = traci.lane.getMaxSpeed(lane)
            t_expected = lane_length / v_max if v_max > 0 else 0.0
            total_delay += max(0.0, t_actual - t_expected) * num_veh

        avg_speed = total_speed / total_vehicles if total_vehicles > 0 else 0.0

        return {
            "total_waiting_time": total_waiting_time,
            "total_queue_length": total_queue_length,
            "total_delay": total_delay,
            "avg_speed": avg_speed,
            "total_vehicles": total_vehicles,
        }

    def _start_sumo(self):
        """
        Start SUMO simulation
        """
        if self.sumo_running:
            traci.close()

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c",
            self.sumo_config,
            "--waiting-time-memory",
            "1000",
            "--no-warnings",
            "--no-step-log",
            "--verbose",
            "false",
        ]

        traci.start(sumo_cmd)
        self.sumo_running = True

    def reset(self, force_restart: bool = False) -> np.ndarray:
        """
        Reset environment to initial state

        Returns:
            Initial observation (state)
        """

        if force_restart:
            if self.sumo_running:
                traci.close()
            self._start_sumo()
        else:
            if not self.sumo_running:
                self._start_sumo()
            else:
                traci.load(
                    [
                        "-c",
                        self.sumo_config,
                        "--waiting-time-memory",
                        "1000",
                        "--no-warnings",
                        "--no-step-log",
                        "--verbose",
                        "false",
                    ]
                )

        # Reset internal state
        self.current_step = 0
        self.time_since_last_phase_change = 0
        self.current_phase_duration = 0
        self.total_phase_switches = 0

        self.cumulative_waiting_time = 0
        self.cumulative_reward = 0

        for _ in range(5):
            traci.simulationStep()

        # Get initial state
        state = self._get_state()

        if self.observation_space_n is None:
            self.observation_space_n = len(state)

        return state

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one time step within the environment

        Args:
            action (int): Action to take (phase index)

        Returns:
            observation: current state
            reward: Reward for the action
            done: Whether episode is finished
            info: Aditional information
        """
        self._apply_action(action)

        for _ in range(self.delta_time):
            traci.simulationStep()
        self.current_step += 1

        self.time_since_last_phase_change += self.delta_time

        state = self._get_state()

        reward = self._compute_reward()
        self.cumulative_reward += reward

        done = self._is_done()

        info = {
            "step": self.current_step,
            "time": traci.simulation.getTime(),
            "cumulative_reward": self.cumulative_reward,
            "cumulative_waiting_time": self.cumulative_waiting_time,
            "total_phase_switches": self.total_phase_switches,
            "current_phase_duration": self.time_since_last_phase_change,
            "current_phase": traci.trafficlight.getPhase(self.tls_id),
        }

        return state, reward, done, info

    def _is_done(self):
        """
        Check if episode is finished

        Returns:
            True if episode should end
        """
        if self.current_step >= self.max_steps:
            print("Max steps reached, ending episode.")
            return True

        if self.end_on_no_vehicles:
            min_expected_vehicles = traci.simulation.getMinExpectedNumber()
            if min_expected_vehicles <= 0:
                return True

        return False

    def close(self):
        """Close the environment and SUMO."""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False


def main():
    tls_id = "7628244053"
    sumoConfig = "./sumo_ingolstadt/simulation/24h_bicycle_sim.sumocfg"
    env = SUMOSimulation(
        tls_id=tls_id,
        use_gui=True,
        max_steps=3600,
        sumo_config=sumoConfig,
        min_green=10,
        max_green=60,
        end_on_no_vehicles=True,
        delta_time=10,
    )

    print(f"Action space size: {env.action_space_n} (0=Continue, 1=Switch)")
    print(f"Observation space size: {env.observation_space_n}")
    print(f"Number of traffic light phases: {env.num_phases}")

    num_episodes = 5

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        step = 0

        print(f"\n=== Episode {episode + 1} ===")

        while not done:
            action = random.randint(0, 1)

            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            step += 1

            if step % 100 == 0:
                action_str = "CONTINUE" if action == 0 else "SWITCH"
                print(
                    f"Step {step}: Action={action_str}, Reward={reward:.2f}, "
                    f"Cumulative={episode_reward:.2f}, Switches={info['total_phase_switches']}"
                )

            state = next_state

        print(f"Episode {episode + 1} finished: Total Reward = {episode_reward:.2f}")
        print(f"Total Waiting Time = {info['cumulative_waiting_time']:.2f}s")
        print(f"Total Phase Switches = {info['total_phase_switches']}")

    env.close()


if __name__ == "__main__":
    main()
