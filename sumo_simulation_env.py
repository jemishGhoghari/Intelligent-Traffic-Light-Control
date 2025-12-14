import random
import numpy as np
import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

# Import traci based on platform
if sys.platform == "linux":
    try:
        import libsumo as traci
    except ImportError:
        import traci
elif sys.platform == "win32":
    import traci
else:
    try:
        import libsumo as traci
    except ImportError:
        import traci


class SUMOSimulation:

    def __init__(
        self,
        tls_id: str,
        sumo_config,
        use_gui: bool = True,
        delta_time: int = 5,
        max_green: int = 60,
        min_green: int = 5,
        yellow_time: int = 3,  # NEW: Yellow phase duration
        reward_type: str = "combined",
        max_steps: int = 3600,
        end_on_no_vehicles: bool = True,
    ):
        self.delta_time = delta_time
        self.tls_id = tls_id
        self.max_green = max_green
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.reward_type = reward_type
        self.sumo_config = sumo_config
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.end_on_no_vehicles = end_on_no_vehicles

        self.current_step = 0
        self.time_since_last_phase_change = 0
        self.current_phase_duration = 0

        # NEW: Track if we're in a yellow phase
        self.in_yellow_phase = False
        self.yellow_phase_remaining = 0
        self.pending_green_phase = None

        self.total_phase_switches = 0
        self.cumulative_waiting_time = 0
        self.cumulative_reward = 0

        # NEW: Track previous metrics for differential rewards
        self.previous_waiting_time = 0
        self.previous_queue_length = 0

        self.sumo_running = False
        self._start_sumo()

        # Action space
        self.action_space_n = 2
        self.observation_space_n = None

        self.num_phases = self._get_num_phases(self.tls_id)

        # NEW: Get green phases only (exclude yellow/red)
        self.green_phases = self._get_green_phases()
        print(
            f"Traffic light has {self.num_phases} total phases, {len(self.green_phases)} green phases"
        )

    def _get_green_phases(self):
        """Get list of green phase indices (exclude yellow/all-red phases)"""
        green_phases = []
        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]

        for i, phase in enumerate(program.phases):
            # A phase is "green" if it contains 'G' or 'g' in its state
            if "G" in phase.state or "g" in phase.state:
                green_phases.append(i)

        return green_phases

    def _get_lane_features(self, lane_id: int):
        """
        Get Lane features like Queue Length, Waiting time of the vehicle, Number of Vehicle on the Lane,
        Mean speed of the vehicles, and delay.

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
        delay = max(0.0, t_actual - t_expected)

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
        """
        Get queue length within given distance from stopline.

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
        lane_features = np.concatenate(lane_features)

        # Extract phase features - use green phases only
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        # One-hot encode green phases
        phase_one_hot = np.zeros(len(self.green_phases), dtype=float)
        if current_phase in self.green_phases:
            phase_idx = self.green_phases.index(current_phase)
            phase_one_hot[phase_idx] = 1.0

        # Normalize time since last change
        time_norm = min(1.0, self.time_since_last_phase_change / self.max_green)

        # NEW: Add flag for whether we can switch now
        can_switch = 1.0 if self.time_since_last_phase_change >= self.min_green else 0.0
        must_switch = (
            1.0 if self.time_since_last_phase_change >= self.max_green else 0.0
        )

        phase_features = np.concatenate(
            [
                phase_one_hot,
                np.array([time_norm, can_switch, must_switch], dtype=float),
            ]
        )

        state = np.concatenate([lane_features, phase_features])
        return state

    def _apply_action(self, action: int):
        """
        Apply the continue/switch action to the traffic light.

        CRITICAL FIX: Properly enforce min_green and max_green constraints!

        Action formulation:
            - action 0: continue current phase
            - action 1: switch to next phase

        Args:
            action (int): 0 (continue) or 1 (switch)
        """
        # If in yellow phase, we can't take actions - just wait
        if self.in_yellow_phase:
            return

        # CRITICAL: Enforce max_green - MUST switch if time exceeded
        if self.time_since_last_phase_change >= self.max_green:
            self._switch_to_next_phase()
            return

        # If action is to switch AND we've met min_green requirement
        if action == 1 and self.time_since_last_phase_change >= self.min_green:
            self._switch_to_next_phase()

        # Otherwise, continue current phase (action 0 or min_green not met)

    def _switch_to_next_phase(self):
        """
        Switch to the next green phase with proper yellow phase transition
        """
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        # Find next green phase
        if current_phase in self.green_phases:
            current_green_idx = self.green_phases.index(current_phase)
            next_green_idx = (current_green_idx + 1) % len(self.green_phases)
            next_green_phase = self.green_phases[next_green_idx]
        else:
            # If somehow not in green phase, go to first green phase
            next_green_phase = self.green_phases[0]

        # Set yellow phase (assuming yellow phases exist between greens)
        # This is traffic-light-specific, might need adjustment
        yellow_phase = current_phase + 1

        # Check if the next phase is indeed a yellow phase
        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        if yellow_phase < len(program.phases):
            phase_state = program.phases[yellow_phase].state
            if "y" in phase_state or "Y" in phase_state:
                # It's a yellow phase, use it
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                self.in_yellow_phase = True
                self.yellow_phase_remaining = self.yellow_time
                self.pending_green_phase = next_green_phase
            else:
                # No yellow phase, switch directly
                traci.trafficlight.setPhase(self.tls_id, next_green_phase)
        else:
            # Out of bounds, switch directly to next green
            traci.trafficlight.setPhase(self.tls_id, next_green_phase)

        self.time_since_last_phase_change = 0
        self.total_phase_switches += 1

    def _update_yellow_phase(self):
        """Handle yellow phase countdown"""
        if self.in_yellow_phase:
            self.yellow_phase_remaining -= self.delta_time

            if self.yellow_phase_remaining <= 0:
                # Yellow phase complete, switch to pending green
                if self.pending_green_phase is not None:
                    traci.trafficlight.setPhase(self.tls_id, self.pending_green_phase)
                    self.pending_green_phase = None

                self.in_yellow_phase = False
                self.yellow_phase_remaining = 0

    def _compute_reward(self) -> float:
        """
        Calculate reward based on traffic metrics.

        CRITICAL FIX: Use differential rewards (change in metrics, not absolute values)
        This gives the agent clearer learning signals!
        """
        metrics = self._calculate_metrics()

        if self.reward_type == "waiting_time":
            # Differential: reward = -change in waiting time
            delta_waiting = metrics["total_waiting_time"] - self.previous_waiting_time
            reward = -delta_waiting / 10.0  # Normalize
            self.previous_waiting_time = metrics["total_waiting_time"]

        elif self.reward_type == "queue":
            # Differential: reward = -change in queue
            delta_queue = metrics["total_queue_length"] - self.previous_queue_length
            reward = -delta_queue
            self.previous_queue_length = metrics["total_queue_length"]

        elif self.reward_type == "delay":
            reward = -metrics["total_delay"] / 10.0

        elif self.reward_type == "combined":
            # Use differential metrics
            delta_waiting = metrics["total_waiting_time"] - self.previous_waiting_time
            delta_queue = metrics["total_queue_length"] - self.previous_queue_length

            # Reward formula: penalize increases, reward decreases
            reward = (
                -0.5 * delta_waiting / 10.0  # Normalized waiting time change
                - 0.5 * delta_queue  # Queue length change
            )

            # Small bonus for throughput (vehicles moving)
            if metrics["total_vehicles"] > 0:
                reward += 0.1 * metrics["avg_speed"] / 15.0  # Normalized speed bonus

            # Small penalty for excessive switching
            if (
                self.total_phase_switches > 0
                and self.time_since_last_phase_change < self.min_green + 2
            ):
                reward -= 0.2  # Penalize rapid switching

            self.previous_waiting_time = metrics["total_waiting_time"]
            self.previous_queue_length = metrics["total_queue_length"]
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        self.cumulative_waiting_time = metrics["total_waiting_time"]

        # Clip reward to reasonable range
        reward = np.clip(reward, -10.0, 10.0)

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
            self.sumo_running = False

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
            "--scale",
            "5.0",
        ]

        traci.start(sumo_cmd)
        self.sumo_running = True

    def reset(self, force_restart: bool = False) -> np.ndarray:
        """
        Reset environment to initial state

        Returns:
            Initial observation (state)
        """

        if force_restart or not self.sumo_running:
            if self.sumo_running:
                traci.close()
                self.sumo_running = False
            self._start_sumo()
        else:
            # Load new simulation without restarting SUMO
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
                    "--scale",
                    "5.0",
                ]
            )

        # Reset internal state
        self.current_step = 0
        self.time_since_last_phase_change = 0
        self.current_phase_duration = 0
        self.total_phase_switches = 0
        self.cumulative_waiting_time = 0
        self.cumulative_reward = 0

        # Reset yellow phase tracking
        self.in_yellow_phase = False
        self.yellow_phase_remaining = 0
        self.pending_green_phase = None

        # Reset differential reward tracking
        self.previous_waiting_time = 0
        self.previous_queue_length = 0

        # Warm-up simulation
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
            action (int): Action to take (0=continue, 1=switch)

        Returns:
            observation: current state
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        # Apply action (will be ignored if in yellow phase)
        self._apply_action(action)

        # Simulate for delta_time steps
        for _ in range(self.delta_time):
            traci.simulationStep()
            self._update_yellow_phase()  # Update yellow phase countdown

        self.current_step += 1

        # Only increment time_since_last_phase_change if not in yellow
        if not self.in_yellow_phase:
            self.time_since_last_phase_change += self.delta_time

        state = self._get_state()
        reward = self._compute_reward()
        self.cumulative_reward += reward
        done = self._is_done()

        current_metrics = self._calculate_metrics()

        info = {
            "step": self.current_step,
            "time": traci.simulation.getTime(),
            "cumulative_reward": self.cumulative_reward,
            "cumulative_waiting_time": self.cumulative_waiting_time,
            "total_phase_switches": self.total_phase_switches,
            "current_phase_duration": self.time_since_last_phase_change,
            "current_phase": traci.trafficlight.getPhase(self.tls_id),
            "in_yellow": self.in_yellow_phase,
            "current_queue_length": current_metrics["total_queue_length"],
            "current_waiting_time": current_metrics["total_waiting_time"],
            "current_vehicles": current_metrics["total_vehicles"],
            "avg_speed": current_metrics["avg_speed"],
        }

        return state, reward, done, info

    def _is_done(self):
        """
        Check if episode is finished

        Returns:
            True if episode should end
        """
        if self.current_step >= self.max_steps:
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
        reward_type="combined",
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
