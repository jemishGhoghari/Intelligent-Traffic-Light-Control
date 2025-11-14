import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import libsumo as traci

sumoBinary = "sumo"  # or "sumo-gui" for graphical version
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
