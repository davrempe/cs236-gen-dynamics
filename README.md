# CS236 Final Project: Generating Physical Dynamics of 3D Rigid Objects
### Davis Rempe
This repo contains all code for the final project. Please see the `video_examples` for sampled trajectories from various versions of the proposed model.

The main files are:
* `step_dataset.py` - data loader for simulation datasets.
* `step_train.py` - script to train model.
* `step_test.py` - script to evaluate model (`step_test_anomaly.py` is modified version to evaluate classification based on likelihood of a trajectory and `step_test_single.py` is modified version to evaluate the MLP recurrent module baselinei).

All other directories contain a readme describing the contained files.
