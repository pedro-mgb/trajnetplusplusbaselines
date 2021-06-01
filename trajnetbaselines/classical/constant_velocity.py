import numpy as np
import numpy.linalg
import trajnetplusplustools


def predict(input_paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    curr_position = xy[obs_length - 1]
    curr_velocity = xy[obs_length - 1] - xy[obs_length - 2]
    output_rel_scenes = np.array([i * curr_velocity for i in range(1, n_predict + 1)])
    output_scenes = curr_position + output_rel_scenes

    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    return multimodal_outputs


def predict_prior2(input_paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}
    pred_length = n_predict

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    curr_position = xy[obs_length - 1]
    curr_velocity = xy[obs_length - 1] - xy[obs_length - 2]
    curr_velocity_prev = xy[obs_length - 2] - xy[obs_length - 3]
    output_vel = np.array([curr_velocity_prev if i % 2 == 1 else curr_velocity for i in range(1, n_predict + 1)])
    output_scenes = curr_position + np.array([np.sum(output_vel[:j], axis=0) for j in range(1, n_predict + 1)])

    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    return multimodal_outputs


def predict_var_angle(input_paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}
    pred_length = n_predict

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    curr_position = xy[obs_length - 1]
    all_velocities = xy[1:obs_length] - xy[:obs_length - 1]
    curr_velocity = xy[obs_length - 1] - xy[obs_length - 2]
    curr_velocity_prev = xy[obs_length - 2] - xy[obs_length - 3]
    velocity_norm = np.linalg.norm(curr_velocity, axis=1)
    velocity_norm = np.repeat(np.expand_dims(velocity_norm, 1), 2, axis=1)
    angle_vel = np.arctan2(curr_velocity[:, 1], curr_velocity[:, 0])
    angle_vel_prev = np.arctan2(curr_velocity_prev[:, 1], curr_velocity_prev[:, 0])
    angle_var = angle_vel - angle_vel_prev
    output_vel = np.array([velocity_norm * np.concatenate((np.expand_dims(np.cos(angle_vel + angle_var * i), axis=1),
                                                           np.expand_dims(np.sin(angle_vel + angle_var * i), axis=1)),
                                                          axis=1) for i in range(1, n_predict + 1)])
    output_scenes = curr_position + np.array([np.sum(output_vel[:j], axis=0) for j in range(1, n_predict + 1)])

    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    return multimodal_outputs


def predict_var_angle_complex(input_paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}
    pred_length = n_predict

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    curr_position = xy[obs_length - 1]
    all_velocities = xy[1:obs_length] - xy[:obs_length - 1]
    all_velocity_angles = np.arctan2(all_velocities[:, :, 1], all_velocities[:, :, 0])
    num_vels = all_velocity_angles.shape[0]  # length in time of the velocities
    angles_half1, angles_half2 = all_velocity_angles[:int(num_vels/2)], all_velocity_angles[int(num_vels/2):]
    avg_angles_half1, avg_angles_half2 = np.mean(angles_half1, axis=0), np.mean(angles_half2, axis=0)
    output_scenes = np.full((n_predict, curr_position.shape[0], curr_position.shape[1]), float('nan'))
    for person in range(xy.shape[1]):
        if avg_angles_half1[person] * avg_angles_half2[person] < 0:
            print("FLIPPING")
        else:
            print("NOT FLIPPING")
    """
    velocity_norm = np.linalg.norm(curr_velocity, axis=1)
    velocity_norm_prev = np.linalg.norm(curr_velocity_prev, axis=1)
    # angle of the velocities (counter clockwise) - in radians
    angle_vel = np.arctan2(curr_velocity[:, 1], curr_velocity[:, 0])
    angle_vel_prev = np.arctan2(curr_velocity_prev[:, 1], curr_velocity_prev[:, 0])
    angle_var = angle_vel - angle_vel_prev
    all_velocity_angles = np.arctan2(all_velocities[:, :, 1], all_velocities[:, :, 0])
    angle_var = np.mean(all_velocity_angles[-2:, :] - all_velocity_angles[-3:-1, :], axis=0)
    velocity_norm = np.mean(np.linalg.norm(all_velocities[-2:, :, :], axis=2), axis=0)
    velocity_norm = np.repeat(np.expand_dims(velocity_norm, 1), 2, axis=1)
    output_vel = np.array([velocity_norm * np.concatenate((np.expand_dims(np.cos(angle_vel + angle_var * i), axis=1),
                                                           np.expand_dims(np.sin(angle_vel + angle_var * i), axis=1)),
                                                          axis=1) for i in range(1, n_predict + 1)])
    """
    all_velocities = xy[1:obs_length] - xy[:obs_length - 1]
    curr_velocity = xy[obs_length - 1] - xy[obs_length - 2]
    curr_velocity_prev = xy[obs_length - 2] - xy[obs_length - 3]
    velocity_norm = np.linalg.norm(curr_velocity, axis=1)
    velocity_norm = np.repeat(np.expand_dims(velocity_norm, 1), 2, axis=1)
    angle_vel = np.arctan2(curr_velocity[:, 1], curr_velocity[:, 0])
    angle_vel_prev = np.arctan2(curr_velocity_prev[:, 1], curr_velocity_prev[:, 0])
    angle_var = angle_vel - angle_vel_prev
    output_vel = np.array([velocity_norm * np.concatenate((np.expand_dims(np.cos(angle_vel + angle_var * i), axis=1),
                                                           np.expand_dims(np.sin(angle_vel + angle_var * i), axis=1)),
                                                          axis=1) for i in range(1, n_predict + 1)])
    output_scenes = curr_position + np.array([np.sum(output_vel[:j], axis=0) for j in range(1, n_predict + 1)])

    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    return multimodal_outputs
