import numpy as np
import trajnetplusplustools


def predict(paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}

    xy = trajnetplusplustools.Reader.paths_to_xy(paths)
    curr_position = xy[obs_length - 1]
    output_scenes = np.repeat(np.expand_dims(curr_position, axis=0), n_predict, axis=0)

    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    return multimodal_outputs
