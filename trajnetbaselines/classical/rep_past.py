import numpy as np
import numpy.linalg
import trajnetplusplustools


def predict(input_paths, predict_all=True, n_predict=12, obs_length=9, num_past=3):
    multimodal_outputs = {}

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    xy = xy[:obs_length, :, :]  # only observed trajectory
    curr_position = xy[-1]
    velocities = xy[(obs_length - num_past):] - xy[(obs_length - num_past - 1):-1]
    output_vel = np.array([velocities[-((i % num_past) + 1)] for i in range(0, n_predict)])
    # output_vel = np.array([curr_velocity_prev if i % 2 == 1 else curr_velocity for i in range(1, n_predict + 1)])
    output_scenes = curr_position + np.array([np.sum(output_vel[:j], axis=0) for j in range(1, n_predict + 1)])

    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    return multimodal_outputs


def predict_sym(input_paths, predict_all=True, n_predict=12, obs_length=9, num_past=8):
    multimodal_outputs = {}

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    xy = xy[:obs_length, :, :]  # only observed trajectory
    curr_position = xy[-1]
    curr_velocity = xy[-1] - xy[-2]
    velocities = xy[(obs_length - num_past):] - xy[(obs_length - num_past - 1):-1]
    speeds = np.repeat(np.expand_dims(np.linalg.norm(velocities, axis=2), axis=2), 2, axis=2)
    velocity_angles = - np.arctan2(velocities[:, :, 1], velocities[:, :, 0])  # symmetrical
    """
    speeds_mean = np.abs(np.mean(velocities, axis=0))
    # opposite preferred direction (x/y) of speed to perform symmetrical operation (across the more constant direction)
    ax = np.argmin(speeds_mean, axis=1)
    # symmetric velocities along the ax (x or y) component
    contrib = np.ones_like(velocities[0, :, :])
    contrib[:, ax] = -1
    output_vel = np.array([velocities[-((i % num_past) + 1)]*contrib for i in range(0, n_predict)])
    """
    f = lambda idx: -((idx % num_past) + 1)  # get the current index
    p = lambda angle: np.concatenate((np.expand_dims(np.cos(angle), axis=1),
                                      np.expand_dims(np.sin(angle), axis=1)), axis=1)  # convert to euclidean coords
    output_vel = np.array([speeds[f(i - (n_predict - obs_length))] * p(velocity_angles[f(i - (n_predict + obs_length))])
                           if i + obs_length > n_predict else curr_velocity for i in range(0, n_predict)])

    output_scenes = curr_position + np.array([np.sum(output_vel[:j], axis=0) for j in range(1, n_predict + 1)])
    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]
    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs
    return multimodal_outputs


def predict_choice(input_paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}
    pred_length = n_predict

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    curr_position = xy[obs_length - 1]
    all_velocities = xy[1:obs_length] - xy[:obs_length - 1]
    all_velocities_mean = np.mean(all_velocities, axis=0)
    all_velocity_angles = np.arctan2(all_velocities[:, :, 1], all_velocities[:, :, 0])
    num_vels = all_velocity_angles.shape[0]  # length in time of the velocities
    angles_half1, angles_half2 = all_velocity_angles[:int(num_vels / 2)], all_velocity_angles[int(num_vels / 2):]
    avg_angles_half1, avg_angles_half2 = np.mean(angles_half1, axis=0), np.mean(angles_half2, axis=0)
    output_scenes = np.full((n_predict, curr_position.shape[0], curr_position.shape[1]), float('nan'))
    rp_pred = predict(input_paths, predict_all, n_predict, obs_length)
    prim, neigh = rp_pred[0]
    rp_pred = np.concatenate((np.expand_dims(prim, axis=1), neigh), axis=1)
    rp_sym_pred = predict_sym(input_paths, predict_all, n_predict, obs_length)
    prim, neigh = rp_sym_pred[0]
    rp_sym_pred = np.concatenate((np.expand_dims(prim, axis=1), neigh), axis=1)
    for person in range(xy.shape[1]):
        range_distance_xy = np.abs(all_velocities_mean[person, 0] / all_velocities_mean[person, 1])
        range_angles = np.abs(avg_angles_half2[person] / avg_angles_half1[person])
        if range_angles > 2.0 or range_angles < 0.5:  # TODO fix this condition?
            """
            and \
                (range_distance_xy > 2 or range_distance_xy < 0.5):"""
            print("SYM")
            #  an extra path is sent because models expect more than one
            output_scenes[:, person, :] = rp_sym_pred[:, person, :]
        else:
            print("Regular")
            #  an extra path is sent because models expect more than one
            output_scenes[:, person, :] = rp_pred[:, person, :]
    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]
    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs
    return multimodal_outputs
