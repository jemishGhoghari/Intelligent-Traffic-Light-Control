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


sumoBinary = "sumo-gui"  # or "sumo-gui" for graphical version
sumoConfig = "./sumo_ingolstadt/simulation/24h_bicycle_sim.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]


class SUMOSimulation:
    def __init__(
        self,
        tls_id: str,
        delta_time: int = 5,
        max_green: int = 60,
        min_green: int = 5,
        reward_type: str = "combined",
    ):
        self.queue_length = 0
        self.waiting_time = 0
        self.num_vehicles = 0
        self.mean_speed = 0
        self.edge_id = 0
        self.t_actual = 0
        self.t_expected = 0
        self.delay = 0

        self.delta_time = delta_time
        self.tls_id = tls_id
        self.max_green = max_green
        self.min_green = min_green
        self.reward_type = reward_type

        self.current_phase_duration = 0
        self.time_since_last_phase_change = 0
        self.total_phase_switches = 0
        self.cumulative_waiting_time = 0

        self.num_phases = self._get_num_phases(self.tls_id)

    def _get_lane_features(self, lane_id: int):
        """
        Get Lane features like Queue Length, Waiting time of the vehicle, Number of Vehicle on the Lane,
        Mean speed of the vehcles, and delay.

        Args:
            lane_id (int): lane id of the incoming or outgoing edge
        """

        # Get total queue length at the stopline of specific lane
        self.queue_length = self._get_queue_length_near_stopline(
            lane_id, distance_from_stop=50
        )

        # Calculate Vehicle Waiting times and Number of Vehicles.
        self.waiting_time = traci.lane.getWaitingTime(lane_id)
        self.num_vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
        self.mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)

        # Delay: Calculate the delay -> realistic travel time - expected travel time
        self.edge_id = traci.lane.getEdgeID(lane_id)
        self.t_actual = traci.edge.getTraveltime(self.edge_id)
        lane_length = traci.lane.getLength(lane_id)
        v_max = traci.lane.getMaxSpeed(lane_id)
        self.t_expected = lane_length / v_max if v_max > 0 else 0.0
        self.delay = max(0.0, self.t_actual - self.t_expected)  # [s]

        return np.array(
            [
                self.queue_length,
                self.waiting_time,
                self.num_vehicles,
                self.mean_speed,
                self.delay,
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
        self.current_phase_duration += self.delta_time

        if action == 0:
            if self.current_phase_duration < self.max_green:
                pass
            else:
                self._switch_to_next_phase()
        elif action == 1:
            if self.current_phase_duration >= self.min_green:
                self._switch_to_next_phase()

    def _switch_to_next_phase(self):
        """
        Switch to the next phase in the traffic light cycle
        """
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        next_phase = (current_phase + 1) % self.num_phases

        traci.trafficlight.setPhase(self.tls_id, next_phase)

        self.current_phase_duration = 0
        self.time_since_last_phase_change = 0
        self.total_phase_switches = 0

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
            total_queue_length += self._get_queue_length_near_stopline(lane, 60)
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


def main():
    traci.start(sumoCmd)
    step = 0

    tls_id = "7628244053"
    env = SUMOSimulation(tls_id)

    try:
        while True:  # Run for 1000 simulation steps
            traci.simulationStep()

            state = env._get_state()
            # print(state)
            step += 1
    except KeyboardInterrupt:
        traci.close()
        sys.exit()


if __name__ == "__main__":
    main()
