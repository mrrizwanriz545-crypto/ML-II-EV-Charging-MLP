# EV Charging Station Load Prediction using Multi-Layer Perceptron (MLP)
**Course:** AIC205A – Machine Learning-II | Ramaiah University of Applied Sciences

## Overview
This project implements a feed-forward Multi-Layer Perceptron (`9 -> 128 -> 64 -> 32 -> 1`, ReLU activation, Adam optimizer) to forecast hourly EV charging station power demand (kW) and classify grid dispatch tiers (`Low`, `Medium`, `Peak`).

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the prototype: `python ev_charging_mlp.py`