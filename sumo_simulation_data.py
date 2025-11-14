import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import libsumo as traci
import numpy as np
import random
import timeit

# sumoBinary = "sumo"  # or "sumo-gui" for graphical version
# sumoConfig = "./sumo_ingolstadt/simulation/24h_sim.sumocfg"
# sumoCmd = [sumoBinary, "-c", sumoConfig]


# def main():
#     traci.start(sumoCmd)

#     step = 0

#     try:
#         while True:  # Run for 1000 simulation steps
#             traci.simulationStep()

#             vehicle_ids = traci.vehicle.getIDList()
#             print(
#                 f"Step {step}: Number of vehicles in the simulation: {len(vehicle_ids)}"
#             )
#             step += 1
#     except KeyboardInterrupt:
#         traci.close()
#         sys.exit()


# if __name__ == "__main__":
#     main()


##################################################################################################
##########:= SUMO Simulation setup for the traffic light state and traffic simulation =:##########
##################################################################################################
class SumoSimulationData:
    def __init__(
        self,
        Model,
        Memory,
        sumoCmd,
        gamma,
        max_steps,
        green_duration,
        yellow_duration,
        num_states,
        num_actions,
        training_epochs,
    ):
        self._Model = Model
        self._Memory = Memory
        self._sumoCmd = sumoCmd
        self._gamma = gamma
        self._step = 0
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._training_epochs = training_epochs
        self._reward_store = []
        self._cumulative_wait_store = []
        self._queue_length_store = []

    def run_simulation(self, episode, epsilon):
        start_time = timeit.default_timer()
        traci.start(self._sumoCmd)

        print("Start simulation...")

        self._step = 0
        self._waiting_times = {}
        self._sum_waiting_time = 0
        self._sum_negative_reward = 0
        self._sum_queue_length = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        # while self._step < self._max_steps:
