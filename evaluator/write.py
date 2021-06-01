import shutil
import os
import pickle

import torch
import numpy as np

import trajnetplusplustools
import trajnetbaselines

## Parallel Compute
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


def process_scene(predictor, model_name, paths, scene_goal, args):
    ## For each scene, get predictions
    if 'sf_opt' in model_name:
        predictions = predictor(paths, sf_params=[0.5, 5.0, 0.3], n_predict=args.pred_length,
                                obs_length=args.obs_length)  ## optimal sf_params (no collision constraint) [0.5, 1.0, 0.1],
    elif 'orca_opt' in model_name:
        predictions = predictor(paths, orca_params=[0.4, 1.0, 0.3], n_predict=args.pred_length,
                                obs_length=args.obs_length)  ## optimal orca_params (no collision constraint) [0.25, 1.0, 0.3]
    elif ('sf' in model_name) or ('orca' in model_name) or ('kf' in model_name) or ('cv' in model_name) or \
            ('cs_va' in model_name) or ('cp' in model_name) or ('ca' in model_name) or ('aftp' in model_name) or \
            ('ls_cf' in model_name) or ('rp' in model_name) or ('rp_sym' in model_name) or ('rp_choice' in model_name):
        predictions = predictor(paths, n_predict=args.pred_length, obs_length=args.obs_length)
    else:
        predictions = predictor(paths, scene_goal, n_predict=args.pred_length, obs_length=args.obs_length,
                                modes=args.modes, args=args)
    return predictions


def main(args=None):
    ## List of .json file inside the args.path (waiting to be predicted by the testing model)
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) if
                       not f.startswith('.') and f.endswith('.ndjson')])
    all_goals = {}
    seq_length = args.obs_length + args.pred_length

    ## Handcrafted Baselines (if included)
    if args.kf:
        args.output.append(os.path.sep + 'kf.pkl')
    if args.sf:
        args.output.append(os.path.sep + 'sf.pkl')
        args.output.append(os.path.sep + 'sf_opt.pkl')
    if args.orca:
        args.output.append(os.path.sep + 'orca.pkl')
        args.output.append(os.path.sep + 'orca_opt.pkl')
    if args.cv:
        args.output.append(os.path.sep + 'cv.pkl')
    if args.cs_va:
        args.output.append(os.path.sep + 'cs_va.pkl')
    if args.rp:
        args.output.append(os.path.sep + 'rp.pkl')
    if args.rp_s:
        args.output.append(os.path.sep + 'rp_sym.pkl')
    if args.rp_c:
        args.output.append(os.path.sep + 'rp_choice.pkl')
    if args.ls_cf:
        args.output.append(os.path.sep + 'ls_cf.pkl')
    if args.cv2:
        args.output.append(os.path.sep + 'cv2.pkl')
    if args.cp:
        args.output.append(os.path.sep + 'cp.pkl')
    if args.ca:
        args.output.append(os.path.sep + 'ca.pkl')
    if args.aftp:
        args.output.append(os.path.sep + 'aftp.pkl')

    ## Extract Model names from arguments and create its own folder in 'test_pred' for storing predictions
    ## WARNING: If Model predictions already exist from previous run, this process SKIPS WRITING
    for model in args.output:
        model_name = model.split(os.path.sep)[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Check if model predictions already exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        if not os.path.exists(args.path + model_name):
            os.makedirs(args.path + model_name)
        else:
            print('Predictions corresponding to {} already exist.'.format(model_name))
            print('Loading the saved predictions')
            continue

        ## Start writing predictions in dataset/test_pred
        for dataset in datasets:
            # Model's name
            name = dataset.replace(args.path.replace('_pred', '') + 'test' + os.path.sep, '') + '.ndjson'
            print('NAME: ', name)

            # Loading the APPROPRIATE model
            ## Keep Adding Different Model Architectures to this List
            print("Model Name: ", model_name)
            goal_flag = False
            if 'kf' in model_name:
                print("Kalman")
                predictor = trajnetbaselines.classical.kalman.predict
            elif 'sf' in model_name:
                print("Social Force")
                predictor = trajnetbaselines.classical.socialforce.predict
            elif 'orca' in model_name:
                print("ORCA")
                predictor = trajnetbaselines.classical.orca.predict
            elif 'rp' in model_name:
                if 'sym' in model_name:
                    print("Model that Replicates Past, with Symmetry")
                    predictor = trajnetbaselines.classical.rep_past.predict_sym
                elif 'choice' in model_name:
                    print("Model that Replicates Past, with CHOICE")
                    predictor = trajnetbaselines.classical.rep_past.predict_choice
                else:
                    print("Model that Replicates Past")
                    predictor = trajnetbaselines.classical.rep_past.predict
            elif 'cs_va' in model_name:
                print("CS-VA (Constant Speed Varying Angle)")
                predictor = trajnetbaselines.classical.constant_velocity.predict_var_angle
            elif 'ls_cf' in model_name:
                print("LSCF (Least Squares Circle Fit) based model")
                predictor = trajnetbaselines.classical.ls_circle.predict
            elif 'cv2' in model_name:
                print("CV2 (using past 2 positions)")
                predictor = trajnetbaselines.classical.constant_velocity.predict_prior2
            elif 'cv' in model_name:
                print("CV")
                predictor = trajnetbaselines.classical.constant_velocity.predict
            elif 'cp' in model_name:
                print("CP (Constant Position)")
                predictor = trajnetbaselines.classical.constant_position.predict
            elif 'ca' in model_name:
                print("CA (Constant Acceleration)")
                predictor = trajnetbaselines.classical.constant_acceleration.predict
            elif 'aftp' in model_name:
                print("AFTP")
                predictor = trajnetbaselines.classical.aftp.predict
            elif 'sgan' in model_name:
                print("SGAN")
                predictor = trajnetbaselines.sgan.SGANPredictor.load(model)
                device = torch.device('cpu')
                predictor.model.to(device)
                goal_flag = predictor.model.generator.goal_flag
            elif 'vae' in model_name:
                print("VAE")
                predictor = trajnetbaselines.vae.VAEPredictor.load(model)
                device = torch.device('cpu')
                predictor.model.to(device)
                goal_flag = predictor.model.goal_flag
            elif 'lstm' in model_name:
                print("LSTM")
                predictor = trajnetbaselines.lstm.LSTMPredictor.load(model)
                device = torch.device('cpu')
                predictor.model.to(device)
                goal_flag = predictor.model.goal_flag
            else:
                print("Model Architecture not recognized")
                raise ValueError

            # Read Scenes from 'test' folder
            reader = trajnetplusplustools.Reader(args.path.replace('_pred', '') + dataset + '.ndjson',
                                                 scene_type='paths')
            ## Necessary modification of train scene to add filename (for goals)
            scenes = [(dataset, s_id, s) for s_id, s in reader.scenes()]

            ## Consider goals
            ## Goal file must be present in 'goal_files/test_private' folder
            ## Goal file must have the same name as corresponding test file
            if goal_flag:
                print("Loading Test Goals file")
                goal_dict = pickle.load(open('goal_files' + os.path.sep + 'test_private' + os.path.sep + dataset +
                                             '.pkl', "rb"))
                all_goals[dataset] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scenes}

            ## Get Goals
            if goal_flag:
                scene_goals = [np.array(all_goals[filename][scene_id]) for filename, scene_id, _ in scenes]
            else:
                scene_goals = [np.zeros((len(paths), 2)) for _, scene_id, paths in scenes]

            # Get the model prediction and write them in corresponding test_pred file
            # VERY IMPORTANT: Prediction Format
            # The predictor function should output a dictionary. The keys of the dictionary should correspond to the prediction modes.
            # ie. predictions[0] corresponds to the first mode. predictions[m] corresponds to the m^th mode.... Multimodal predictions!
            # Each modal prediction comprises of primary prediction and neighbour (surrrounding) predictions i.e. predictions[m] = [primary_prediction, neigh_predictions]
            # Note: Return [primary_prediction, []] if model does not provide neighbour predictions
            # Shape of primary_prediction: Tensor of Shape (Prediction length, 2)
            # Shape of Neighbour_prediction: Tensor of Shape (Prediction length, n_tracks - 1, 2).
            # (See LSTMPredictor.py for more details)
            scenes = tqdm(scenes)
            with open(args.path + '{}{}{}'.format(model_name, os.path.sep, name), "a") as myfile:
                ## Get all predictions in parallel. Faster! Original number of processes was 12, but was reduced so it
                #   works on a higher number of machines (e.g. ones running Windows OS)
                if args.no_parallel:
                    pred_list = [process_scene(predictor, model_name, paths, scene_goal, args)
                                 for (_, _, paths), scene_goal in zip(scenes, scene_goals)]
                else:
                    pred_list = Parallel(n_jobs=4)(
                        delayed(process_scene)(predictor, model_name, paths, scene_goal, args)
                        for (_, _, paths), scene_goal in zip(scenes, scene_goals))

                ## Write All Predictions
                for (predictions, (_, scene_id, paths)) in zip(pred_list, scenes):
                    ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
                    observed_path = paths[0]
                    frame_diff = observed_path[1].frame - observed_path[0].frame
                    first_frame = observed_path[args.obs_length - 1].frame + frame_diff
                    ped_id = observed_path[0].pedestrian
                    ped_id_ = []
                    for j, _ in enumerate(paths[1:]):  ## Only need neighbour ids
                        ped_id_.append(paths[j + 1][0].pedestrian)

                    ## Write SceneRow
                    scenerow = trajnetplusplustools.SceneRow(scene_id, ped_id, observed_path[0].frame,
                                                             observed_path[0].frame + (seq_length - 1) * frame_diff,
                                                             2.5, 0)
                    # scenerow = trajnetplusplustools.SceneRow(scenerow.scene, scenerow.pedestrian, scenerow.start, scenerow.end, 2.5, 0)
                    myfile.write(trajnetplusplustools.writers.trajnet(scenerow))
                    myfile.write('\n')

                    for m in range(len(predictions)):
                        prediction, neigh_predictions = predictions[m]
                        ## Write Primary
                        for i in range(len(prediction)):
                            track = trajnetplusplustools.TrackRow(first_frame + i * frame_diff, ped_id,
                                                                  prediction[i, 0].item(), prediction[i, 1].item(), m,
                                                                  scene_id)
                            myfile.write(trajnetplusplustools.writers.trajnet(track))
                            myfile.write('\n')

                        ## Write Neighbours (if non-empty)
                        if len(neigh_predictions):
                            for n in range(neigh_predictions.shape[1]):
                                neigh = neigh_predictions[:, n]
                                for j in range(len(neigh)):
                                    track = trajnetplusplustools.TrackRow(first_frame + j * frame_diff, ped_id_[n],
                                                                          neigh[j, 0].item(), neigh[j, 1].item(), m,
                                                                          scene_id)
                                    myfile.write(trajnetplusplustools.writers.trajnet(track))
                                    myfile.write('\n')
        print('')


if __name__ == '__main__':
    main()
