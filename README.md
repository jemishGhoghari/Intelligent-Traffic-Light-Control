# Intelligent-Traffic-Light-Control

This repository contains the research and implementation of the traffic control at a intersaction using state of art deep learning model.

## Setup:

First, Open a fresh terminal and execute following commnads to setup SUMO simulation and repository.

* **Install SUMO Simulator:**
  `sudo apt-get install sumo sumo-tools sumo-doc`
* **clone github repository:**
  `git clone https://github.com/jemishGhoghari/Intelligent-Traffic-Light-Control.git`
* **Navigate to repository directory and update submodules:**
  `git submodule update --init --recursive`
* **Generate a python environment to install SUMO Traci and Libsumo library:**
  `python3 -m venv venv`
* **activate python environment**
  `source venv/bin/activate`
* **Install SUMO traci and libsumo**
  `pip3 install traci libsumo`
