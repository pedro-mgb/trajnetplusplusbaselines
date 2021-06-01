"""
AFTP - Arc-Fitting and Tangent-Prediction
Can be seen as a slightly more complex geometric model when compared to constant velocity

Original paper: "Neither Too Much nor Too Little: Leveraging Moderate Data in Pedestrian Trajectory Prediction"
[LINK] https://ieeexplore.ieee.org/document/9361154
"""
import numpy as np
import trajnetplusplustools

from .constant_position import predict as cp_predict
from .constant_velocity import predict as cv_predict


def predict(input_paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}
    pred_length = n_predict

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    p2, p1, p0 = xy[obs_length - 1], xy[obs_length - 2], xy[obs_length - 3]
    xc, yc, _ = find_circle(p2[:, 0], p2[:, 1], p1[:, 1], p1[:, 1], p0[:, 0], p0[:, 1])
    kr = (p2[:, 1] - yc) / (p2[:, 0] - xc)
    kl = - 1 / kr
    curr_speed = np.linalg.norm(p2 - p1, axis=1)
    displacement = np.zeros_like(p2)
    displacement[:, 0] = np.sign(p2[:, 0] - p1[:, 0]) * curr_speed / np.sqrt(pow(kl, 2) + 1)
    displacement[:, 1] = np.sign(p2[:, 0] - p1[:, 0]) * curr_speed * kl / np.sqrt(pow(kl, 2) + 1)
    output_rel_scenes = np.array([i * displacement for i in range(1, n_predict + 1)])
    output_scenes = p2 + output_rel_scenes

    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    cp_outputs = cp_predict(input_paths, predict_all, n_predict, obs_length)
    cv_outputs = cv_predict(input_paths, predict_all, n_predict, obs_length)
    cp_primary, cp_neighs = cp_outputs[0]
    cv_primary, cv_neighs = cv_outputs[0]
    for person in range(p2.shape[0]):
        if np.all(p2[person, 1] == p1[person, 1]) or np.all(p2[person, :] == p1[person, :]) \
                or np.all(p2[person, :] == p0[person, :]):
            #   First condition: if y2==y1, the analytical computation of kl does not apply (results in nan).
            # For more info: https://www.computer.org/csdl/proceedings-article/icaice/2020/914600a444/1rCg6NKwyhq
            #   Second and Third conditions: common pedestrian position, don't use those values because the above
            # circle finding won't work

            # SO - Use Alternative method
            # if the p2 and p0 are equal, then most likely pedestrian is stopped
            use_cv = np.any(p2[person, 0] != p0[person, 0])  # if True, constant velocity; if False, constant position
            if person == 0:
                output_primary = cv_primary if use_cv else cp_primary
            else:
                output_neighs[:, person - 1, :] = cv_neighs[:, person - 1, :] if use_cv \
                    else cp_neighs[:, person - 1, :]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs
    return multimodal_outputs


def find_circle(x1, y1, x2, y2, x3, y3):
    """
    https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
    """
    x12 = x1 - x2
    x13 = x1 - x3
    y12 = y1 - y2
    y13 = y1 - y3
    y31 = y3 - y1
    y21 = y2 - y1
    x31 = x3 - x1
    x21 = x2 - x1

    sx13 = pow(x1, 2) - pow(x3, 2)
    sy13 = pow(y1, 2) - pow(y3, 2)
    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = (sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) / (2 * (y31 * x12 - y21 * x13))
    g = (sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) / (2 * (x31 * y12 - x21 * y13))
    c = -pow(x1, 2) - pow(y1, 2) - 2 * g * x1 - 2 * f * y1

    xc, yc = -g, -f
    r = np.sqrt(xc * xc + yc * yc - c)
    return xc, yc, r
