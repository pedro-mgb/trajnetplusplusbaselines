import numpy as np
import numpy.linalg
import trajnetplusplustools
from .constant_velocity import predict as predict_cv


def predict(input_paths, predict_all=True, n_predict=12, obs_length=9):
    """
    Circle fit with Least Squares:
    http://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf
    """
    multimodal_outputs = {}

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    xy_obs = xy[:obs_length, :, :]
    # LEAST SQUARES SOLVER STARTS HERE
    N = xy_obs.shape[0]  # number of points
    # centroids for x and y coordinates
    xy_cent = np.mean(xy_obs, axis=0)
    # subtract centroids to coordinates - problem solved here then moved to regular coords
    uv = xy_obs - np.repeat(np.expand_dims(xy_cent, axis=0), N, axis=0)
    u, v = uv[:, :, 0], uv[:, :, 1]
    uu, vv, uv = u * u, v * v, u * v
    s_uu, s_vv, s_uv = np.sum(uu, axis=0), np.sum(vv, axis=0), np.sum(uv, axis=0)
    s_uuu, s_vvv = np.sum(uu * u, axis=0), np.sum(vv * v, axis=0)
    s_uvv, s_vuu = np.sum(u * vv, axis=0), np.sum(v * uu, axis=0)
    s_a, s_b = 0.5 * (s_vvv + s_vuu), 0.5 * (s_uuu + s_uvv)
    # coordinates of the center of the circle - still needs to sum centroids
    vc = (s_b - s_a * s_uu / s_uv) / (s_uv - s_uu * s_vv / s_uv)
    uc = (s_a - vc * s_vv) / s_uv
    r = np.sqrt(uc * uc + vc * vc + (s_uu + s_vv) / N)
    xc, yc = uc + xy_cent[:, 0], vc + xy_cent[:, 1]
    center = np.concatenate((np.expand_dims(xc, axis=1), np.expand_dims(yc, axis=1)), axis=1)
    err = np.sum((u - np.repeat(np.expand_dims(uc, axis=0), N, axis=0)) +
                 (v - np.repeat(np.expand_dims(vc, axis=0), N, axis=0)) -
                 np.repeat(np.expand_dims(r * r, axis=0), N, axis=0), axis=0)
    # closest points of circle to observed trajectory
    r_expanded = np.repeat(np.repeat(np.expand_dims(r, axis=(0, 2)), N, axis=0), 2, axis=2)
    xy_obs_closest = center + r_expanded * (xy_obs - center) / \
                     np.repeat(np.expand_dims(np.linalg.norm(xy_obs - center, axis=2), axis=2), 2, axis=2)
    # obtain the angle variation between two consecutive points in the circle
    xy_obs_closest_centered = xy_obs_closest - center
    distance_pts = xy_obs_closest_centered[1:, :, :] - xy_obs_closest_centered[:-1, :, :]
    distance_pts_to_real = xy_obs - xy_obs_closest
    angle_pts = np.arctan2(xy_obs_closest_centered[:, :, 1], xy_obs_closest_centered[:, :, 0])
    # angle_pts_dist = np.arctan2(distance_pts[:, :, 1], distance_pts[:, :, 0])
    angle_var = (angle_pts[-1, :] - angle_pts[-2, :])
    # angle_var = np.mean(angle_pts[1:, :] - angle_pts[-1:, :], axis=0)
    r_expanded = np.repeat(np.expand_dims(r, axis=1), 2, axis=1)
    points = center + distance_pts_to_real[-1, :, :] + np.array(
        [r_expanded * np.concatenate((np.expand_dims(np.cos(angle_pts[-1, :] + angle_var * i), axis=1),
                                      np.expand_dims(np.sin(angle_pts[-1, :] + angle_var * i), axis=1)),
                                     axis=1) for i in range(1, n_predict + 1)])

    cv_predictions = predict_cv(input_paths, predict_all, n_predict, obs_length)
    prim_cv, neigh_cv = cv_predictions[0]
    cv_predictions = np.concatenate((np.expand_dims(prim_cv, axis=1), neigh_cv), axis=1)
    for person in range(points.shape[1]):
        if np.any(np.isnan(points[:, person])):
            # TODO replace with cv prediction
            points[:, person, :] = cv_predictions[:, person, :]

    output_primary = points[-n_predict:, 0]
    output_neighs = points[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    return multimodal_outputs
