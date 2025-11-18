import os
import sys


if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

print(sys.platform)

if sys.platform == "linux":
    import libsumo as traci
elif sys.platform == "win32":
    import traci
else:
    print("OS not supported!")

import numpy as np
import random
import timeit

sumoBinary = "sumo-gui"  # or "sumo-gui" for graphical version
sumoConfig = "./sumo_ingolstadt/simulation/24h_sim.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]


def main():
    traci.start(sumoCmd)
    step = 0
    try:
        while True:  # Run for 1000 simulation steps
            traci.simulationStep()

            vehicle_ids = traci.vehicle.getIDList()
            print(
                f"Step {step}: Number of vehicles in the simulation: {len(vehicle_ids)}"
            )
            step += 1
    except KeyboardInterrupt:
        traci.close()
        sys.exit()


if __name__ == "__main__":
    main()


##################################################################################################
##########:= SUMO Simulation setup for the traffic light state and traffic simulation =:##########
##################################################################################################
# class SumoSimulationData:
#     def __init__(
#         self,
#         Model,
#         Memory,
#         sumoCmd,
#         gamma,
#         max_steps,
#         green_duration,
#         yellow_duration,
#         num_states,
#         num_actions,
#         training_epochs,
#     ):
#         self._Model = Model
#         self._Memory = Memory
#         self._sumoCmd = sumoCmd
#         self._gamma = gamma
#         self._step = 0
#         self._max_steps = max_steps
#         self._green_duration = green_duration
#         self._yellow_duration = yellow_duration
#         self._num_states = num_states
#         self._num_actions = num_actions
#         self._training_epochs = training_epochs
#         self._reward_store = []
#         self._cumulative_wait_store = []
#         self._queue_length_store = []

#     def run_simulation(self, episode, epsilon):
#         start_time = timeit.default_timer()
#         traci.start(self._sumoCmd)

#         print("Start simulation...")

#         self._step = 0
#         self._waiting_times = {}
#         self._sum_waiting_time = 0
#         self._sum_negative_reward = 0
#         self._sum_queue_length = 0
#         old_total_wait = 0
#         old_state = -1
#         old_action = -1

#         while self._step < self._max_steps:
#             # current_state = self.
#             traci.simulationStep()
#             self._get_state()
#             self._step += 1

#     def _get_state(self):
#         state = np.zeros(self._num_states)
#         car_list = traci.vehicle.getIDList()

#         for car_id in car_list:
#             lane_pos = traci.vehicle.getLanePosition(car_id)
#             lane_id = traci.vehicle.getLaneID(car_id)
#             lane_pos = 750 - lane_pos  # Invert position to measure from traffic light
#             # print(f"Car ID: {car_id}, Lane ID: {lane_id}, Lane Position: {lane_pos}")

# if __name__ == "__main__":
#     # Example usage
#     sumoBinary = "sumo"  # or "sumo-gui" for graphical version
#     sumoConfig = "./sumo_ingolstadt/simulation/24h_sim.sumocfg"
#     sumoCmd = [sumoBinary, "-c", sumoConfig]

#     simulation = SumoSimulationData(
#         Model=None,
#         Memory=None,
#         sumoCmd=sumoCmd,
#         gamma=0.9,
#         max_steps=1000,
#         green_duration=30,
#         yellow_duration=5,
#         num_states=10,
#         num_actions=4,
#         training_epochs=10,
#     )

#     try:
#         simulation.run_simulation(episode=1, epsilon=0.1)
#     except KeyboardInterrupt:
#         traci.close()
#         sys.exit()
