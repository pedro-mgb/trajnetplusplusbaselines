import numpy as np
import trajnetplusplustools


def predict(paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}

    xy = trajnetplusplustools.Reader.paths_to_xy(paths)
    curr_position = xy[obs_length - 1]
    curr_velocity = xy[obs_length - 1] - xy[obs_length - 2]
    prev_velocity = xy[obs_length - 2] - xy[obs_length - 3]
    curr_acceleration = curr_velocity - prev_velocity
    output_velocity = np.array([curr_acceleration * i for i in range(1, n_predict + 1)])
    output_scenes = curr_position + np.array([np.sum(output_velocity[:j], axis=0) for j in range(1, n_predict + 1)])

    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    return multimodal_outputs
