"""
Fit the data to several objective functions
"""
import numpy as np
from scipy.optimize import curve_fit
import trajnetplusplustools
from ..augmentation import center_scene, inverse_scene


def predict(input_paths, predict_all=True, n_predict=12, obs_length=9, obj='linear', norm=False, max_past=3):
    """
    Circle fit with several kind of objective functions
    Learn more here: https://machinelearningmastery.com/curve-fitting-with-python/
    """
    multimodal_outputs = {}

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    xy = xy[:obs_length, :, :]  # only observed positions
    rotation, center = None, None
    if norm:
        xy, rotation, center = center_scene(xy, obs_length=obs_length, offset=0)
    vel = xy[-1] - xy[-2]
    cv_output = np.array([xy[-1] + i * vel for i in range(1, n_predict + 1)])
    output_scenes = np.full_like(cv_output, float('nan'))
    if 'lin' in obj.lower():
        objective = linear_objective
        min_pts = 2
    elif 'quad' in obj.lower():
        objective = quad_objective
        min_pts = 3
    elif 'sin' in obj.lower():
        objective = sin_objective
        min_pts = 3
    elif 'cos' in obj.lower():
        objective = cos_objective
        min_pts = 3
    else:
        raise Exception('Objective ' + obj + ' not available')
    for i in range(xy.shape[1]):  # for each person
        # for partial trajectories, don't use the instants where there is no trajectories
        mask = ~np.isnan(xy[:, i, 0])
        x_data, y_data = xy[mask, i, 0], xy[mask, i, 1]
        num_pts = x_data.shape[0]
        if num_pts > max_past:
            x_data, y_data = x_data[-max_past:], y_data[-max_past:]
            num_pts = max_past
        # optimal parameter data (covariance not used)
        if np.any(np.isnan(cv_output[:, i, 0])):
            # that trajectory will have nan's in the end instants, no prediction to be made
            continue
        elif num_pts < min_pts:
            output_scenes[:, i, :] = cv_output[:, i, :]
            continue
        popt, _ = curve_fit(objective, x_data, y_data)
        new_y = objective(cv_output[:, i, 0], *popt)
        output_scenes[:, i, 0] = cv_output[:, i, 0]
        output_scenes[:, i, 1] = new_y
    if norm:
        output_scenes = inverse_scene(output_scenes, rotation, center)
    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]
    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs
    return multimodal_outputs


def linear_objective(x, a, b):
    return a * x + b


def quad_objective(x, a, b, c):
    return a * x + b * x ** 2 + c


def sin_objective(x, a, b, f):
    return a * np.sin(2 * np.pi * f * x + b)


def cos_objective(x, a, b, f):
    return a * np.cos(2 * np.pi * f * x + b)
