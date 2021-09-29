from collections import defaultdict
import os

import numpy as np
import matplotlib.pyplot as plt

import torch

def one_cold(i, n):
    """Inverse one-hot encoding."""
    x = torch.ones(n, dtype=torch.bool)
    x[i] = 0
    return x

class GridBasedPooling(torch.nn.Module):
    def __init__(self, cell_side=2.0, n=4, hidden_dim=128, out_dim=None,
                 type_='occupancy', pool_size=1, blur_size=1, front=False,
                 embedding_arch='one_layer', pretrained_pool_encoder=None,
                 constant=0, norm=0, layer_dims=None, latent_dim=16, extra_args=None):
        """
        Pools in a grid of size 'n * cell_side' centred at the ped location
        cell_side: Scalar
            size of each cell in real world
        n: Scalar
            number of cells along one dimension
        out_dim: Scalar
            dimension of resultant interaaction vector
        type_: ('occupancy', 'directional', 'social', 'dir_social')
            type of grid-based pooling
        front: Bool 
            if True, pools neighbours only in the front of pedestrian
        embedding_arch: ('one_layer', 'two_layer', 'three_layer', 'lstm_layer')
            architecture to encoder grid tensor
        pretrained_pool_encoder: None
            autoencoder to reduce dimensionality of grid
        constant: int
            background values of pooling grid
        norm: Scalar 
            normalization scheme of pool grid [Default: None]
        extra_args: dictionary
            contains other arguments not present in original Trajnet++ model, such as the use of arc pooling
        """
        super(GridBasedPooling, self).__init__()

        if extra_args is None:
            self.arc = False
            self.arc_radius = self.arc_angle = self.n_r = self.n_a = None
            self.include_occ = False
        else:
            self.arc = extra_args.type == 'arc'
            if self.arc:
                self.arc_radius, self.arc_angle = extra_args.arc_radius, extra_args.arc_angle
                self.n_r, self.n_a = extra_args.n_r, extra_args.n_a
            else:
                self.arc_radius = self.arc_angle = self.n_r = self.n_a = None
            self.include_occ = extra_args.variable_shape
        self.cell_side = cell_side
        self.n = n
        self.type_ = type_
        self.pool_size = pool_size
        self.blur_size = blur_size

        self.norm_pool = False
        self.front = front
        if self.front:
            self.norm_pool = True
        self.constant = constant
        self.norm = norm
        self.pool_scale = 1.0

        ## Type of pooling
        self.pooling_dim = 1
        if self.type_ == 'directional' or self.arc:
            self.pooling_dim = 2
        if self.type_ == 'social':
            ## Encode hidden-dim into latent-dim vector (faster computation)
            self.hidden_dim_encoding = torch.nn.Linear(hidden_dim, latent_dim)
            self.pooling_dim = latent_dim
        if self.type_ == 'dir_social':
            ## Encode hidden-dim into latent-dim vector (faster computation)
            self.hidden_dim_encoding = torch.nn.Linear(hidden_dim, latent_dim)
            self.pooling_dim = latent_dim + 2

        ## Final Representation Size
        if out_dim is None:
            out_dim = hidden_dim
        self.out_dim = out_dim

        ## Pretrained AE
        self.pretrained_model = pretrained_pool_encoder
        input_dim = None
        if self.pretrained_model is not None:
            input_dim = self.pretrained_model[-1].out_features
            if embedding_arch == 'None':
                self.out_dim = input_dim

        ## Embedding Grid / AE Representation
        self.embedding = None
        self.embedding_arch = embedding_arch
        if self.embedding_arch == 'one_layer':
            self.embedding = self.one_layer(input_dim)
        elif self.embedding_arch == 'two_layer':
            self.embedding = self.two_layer(input_dim, layer_dims)
        elif self.embedding_arch == 'three_layer':
            self.embedding = self.three_layer(input_dim, layer_dims)
        elif self.embedding_arch == 'lstm_layer':
            self.embedding = self.lstm_layer(hidden_dim)

    def forward_grid(self, grid):
        """ Encodes the generated grid tensor
        
        Parameters
        ----------
        grid: [num_tracks, self.pooling_dim, self.n, self.n]
            Generated Grid
        Returns
        -------
        interactor_vector: Tensor [num_tracks, self.out_dim]
        """
        num_tracks = grid.size(0)

        ## Encode grid using pre-trained autoencoder (reduce dimensionality)
        if self.pretrained_model is not None:
            if not isinstance(self.pretrained_model[0], torch.nn.Conv2d):
                grid = grid.view(num_tracks, -1)
            mean, std = grid.mean(), grid.std()
            if std == 0:
                std = 0.03
            grid = (grid - mean) / std
            grid = self.pretrained_model(grid)

        ## Normalize Grid (if necessary)
        grid = grid.view(num_tracks, -1)
        ## Normalization schemes
        if self.norm == 1:
            # "Global Norm"
            mean, std = grid.mean(), grid.std()
            std[std == 0] = 0.09
            grid = (grid - mean) / std
        elif self.norm == 2:
            # "Feature Norm"
            mean, std = grid.mean(dim=0, keepdim=True), grid.std(dim=0, keepdim=True)
            std[std == 0] = 0.1
            grid = (grid - mean) / std
        elif self.norm == 3:
            # "Sample Norm"
            mean, std = grid.mean(dim=1, keepdim=True), grid.std(dim=1, keepdim=True)
            std[std == 0] = 0.1
            grid = (grid - mean) / std

        ## Embed grid
        if self.embedding_arch == 'lstm_layer':
            return self.lstm_forward(grid)

        elif self.embedding:
            return self.embedding(grid)

        return grid

    def forward(self, hidden_state, obs1, obs2):
        ## Make chosen grid
        if self.type_ == 'occupancy':
            grid = self.occupancies(obs1, obs2)
        elif self.type_ == 'directional' or self.arc:
            grid = self.directional(obs1, obs2)
        elif self.type_ == 'social':
            grid = self.social(hidden_state, obs1, obs2)
        elif self.type_ == 'dir_social':
            grid = self.dir_social(hidden_state, obs1, obs2)
        else:
            raise Exception('TYPE NOT AVAILABLE')

        ## Forward Grid
        return self.forward_grid(grid)

    def occupancies(self, obs1, obs2):
        ## Generate the Occupancy Map
        return self.occupancy(obs2, past_obs=obs1)

    def directional(self, obs1, obs2):
        ## Makes the Directional Grid

        num_tracks = obs2.size(0)

        ## if only primary pedestrian present
        if num_tracks == 1:
            return self.arc_pooling(obs2) if self.arc else self.occupancy(obs2, None)

        ## Generate values to input in directional grid tensor (relative velocities in this case) 
        vel = obs2 - obs1
        unfolded = vel.unsqueeze(0).repeat(vel.size(0), 1, 1)
        ## [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        relative = unfolded - vel.unsqueeze(1)
        ## Deleting Diagonal (Ped wrt itself)
        ## [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks-1, 2]
        relative = relative[~torch.eye(num_tracks).bool()].reshape(num_tracks, num_tracks-1, 2)

        if self.arc:
            ## Generate Occupancy Map with ARC
            angle_vel = torch.atan2(vel[:, 1], vel[:, 0])
            return self.arc_pooling(obs2, relative, angle_offsets=angle_vel)
        else:
            ## Generate Occupancy Map with GRID
            return self.occupancy(obs2, relative, past_obs=obs1)

    def social(self, hidden_state, obs1, obs2):
        ## Makes the Social Grid

        num_tracks = obs2.size(0)

        ## if only primary pedestrian present
        if num_tracks == 1:
            return self.occupancy(obs2, None, past_obs=obs1)

        ## Generate values to input in hiddenstate grid tensor (compressed hidden-states in this case) 
        ## [num_tracks, hidden_dim] --> [num_tracks, num_tracks-1, pooling_dim]
        hidden_state_grid = hidden_state.repeat(num_tracks, 1).view(num_tracks, num_tracks, -1)
        hidden_state_grid = hidden_state_grid[~torch.eye(num_tracks).bool()].reshape(num_tracks, num_tracks-1, -1)
        hidden_state_grid = self.hidden_dim_encoding(hidden_state_grid)
        
        ## Generate Occupancy Map
        return self.occupancy(obs2, hidden_state_grid, past_obs=obs1)

    def dir_social(self, hidden_state, obs1, obs2):
        ## Makes the Directional + Social Grid

        num_tracks = obs2.size(0)

        ## if only primary pedestrian present
        if num_tracks == 1:
            return self.occupancy(obs2, None)

        ## Generate values to input in directional grid tensor (relative velocities in this case) 
        vel = obs2 - obs1
        unfolded = vel.unsqueeze(0).repeat(vel.size(0), 1, 1)
        ## [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        relative = unfolded - vel.unsqueeze(1)
        ## Deleting Diagonal (Ped wrt itself)
        ## [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks-1, 2]
        relative = relative[~torch.eye(num_tracks).bool()].reshape(num_tracks, num_tracks-1, 2)

        ## Generate values to input in hiddenstate grid tensor (compressed hidden-states in this case) 
        ## [num_tracks, hidden_dim] --> [num_tracks, num_tracks-1, pooling_dim]
        hidden_state_grid = hidden_state.repeat(num_tracks, 1).view(num_tracks, num_tracks, -1)
        hidden_state_grid = hidden_state_grid[~torch.eye(num_tracks).bool()].reshape(num_tracks, num_tracks-1, -1)
        hidden_state_grid = self.hidden_dim_encoding(hidden_state_grid)

        dir_social_rep = torch.cat([relative, hidden_state_grid], dim=2)

        ## Generate Occupancy Map
        return self.occupancy(obs2, dir_social_rep, past_obs=obs1)

    @staticmethod
    def normalize(relative, obs, past_obs):
        ## Normalize pooling grid along direction of pedestrian motion
        diff = torch.cat([obs[:, 1:] - past_obs[:, 1:], obs[:, 0:1] - past_obs[:, 0:1]], dim=1)
        velocity = np.arctan2(diff[:, 0].clone(), diff[:, 1].clone())
        theta = (np.pi / 2) - velocity
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ## Cleaner?
        relative = torch.stack([torch.einsum('tc,ci->ti', pos_instance, torch.Tensor([[ct[i], st[i]], [-st[i], ct[i]]])) for
                                i, pos_instance in enumerate(relative)], dim=0)
        return relative

    def occupancy(self, obs, other_values=None, past_obs=None):
        """Returns the occupancy map filled with respective attributes.
        A different occupancy map with respect to each pedestrian
        Parameters
        ----------
        obs: Tensor [num_tracks, 2]
            Current x-y positions of all pedestrians, used to construct occupancy map.
        other_values: Tensor [num_tracks, num_tracks-1,  2]
            Attributes (self.pooling_dim) of the neighbours relative to pedestrians, to be filled in the occupancy map
            e.g. Relative velocities of pedestrians
        past_obs: Tensor [num_tracks, 2]
            Previous x-y positions of all pedestrians, used to construct occupancy map.
            Useful for normalizing the grid tensor.
        Returns
        -------
        grid: Tensor [num_tracks, self.pooling_dim, self.n, self.n]
        """
        num_tracks = obs.size(0)

        ##mask unseen
        mask = torch.isnan(obs).any(dim=1)
        obs[mask] = 0

        ## if only primary pedestrian present
        if num_tracks == 1:
            return self.constant*torch.ones(1, self.pooling_dim, self.n, self.n, device=obs.device)

        ## Get relative position
        ## [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        unfolded = obs.unsqueeze(0).repeat(obs.size(0), 1, 1)
        relative = unfolded - obs.unsqueeze(1)
        ## Deleting Diagonal (Ped wrt itself)
        ## [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks-1, 2]
        relative = relative[~torch.eye(num_tracks).bool()].reshape(num_tracks, num_tracks-1, 2)

        ## In case of 'occupancy' pooling
        if other_values is None:
            other_values = torch.ones(num_tracks, num_tracks-1, self.pooling_dim, device=obs.device)

        ## Normalize pooling grid along direction of pedestrian motion
        if self.norm_pool:
            relative = self.normalize(relative, obs, past_obs)

        if self.front:
            oij = (relative / (self.cell_side / self.pool_size) + torch.Tensor([self.n * self.pool_size / 2, 0]))
        else:
            oij = (relative / (self.cell_side / self.pool_size) + self.n * self.pool_size / 2)

        range_violations = torch.sum((oij < 0) + (oij >= self.n * self.pool_size), dim=2)
        range_mask = range_violations == 0

        oij[~range_mask] = 0
        other_values[~range_mask] = self.constant
        oij = oij.long()

        ## Flatten
        oi = oij[:, :, 0] * self.n * self.pool_size + oij[:, :, 1]

        # faster occupancy
        occ = self.constant*torch.ones(num_tracks, self.n**2 * self.pool_size**2, self.pooling_dim, device=obs.device)

        ## Fill occupancy map with attributes
        occ[torch.arange(occ.size(0)).unsqueeze(1), oi] = other_values
        occ = torch.transpose(occ, 1, 2)
        occ_2d = occ.view(num_tracks, -1, self.n * self.pool_size, self.n * self.pool_size)

        if self.blur_size == 1:
            occ_blurred = occ_2d
        else:
            occ_blurred = torch.nn.functional.avg_pool2d(
                occ_2d, self.blur_size, 1, int(self.blur_size / 2), count_include_pad=True)

        occ_summed = torch.nn.functional.lp_pool2d(occ_blurred, 1, self.pool_size)
        # occ_summed = torch.nn.functional.avg_pool2d(occ_blurred, self.pool_size)  # faster?
        return occ_summed

    def arc_pooling(self, obs, other_values=None, angle_offsets=None):
        """

        :param obs:
        :param other_values:
        :param angle_offsets:
        :return:
        """
        num_peds = obs.size(0)
        # mask unseen values (the calling model should take care of this, but this is just a safe-guard)
        obs[torch.any(torch.isnan(obs), dim=1)] = 0
        # if only one pedestrian is present
        if num_peds == 1:
            # return self.constant * torch.ones(1, self.pooling_dim, self.n_r, self.n_a, device=obs.device)
            # note the line below may be used if variable shape LSTM network is employed
            ''', torch.zeros(1, 1, self.n_r, self.n_a, device=obs.device) if self.include_occ else None '''
            return torch.zeros(1, self.pooling_dim, self.n_r, self.n_a, device=obs.device)
        # Get relative position of the pedestrians
        # [num_peds, 2] --> [num_peds, num_peds, 2]
        unfolded = obs.unsqueeze(0).repeat(obs.size(0), 1, 1)
        relative_cart = unfolded - obs.unsqueeze(1)
        # convert to polar coordinates 0<->radius, 1<->angle
        relative = torch.cat((torch.norm(relative_cart, dim=2, keepdim=True),
                              torch.atan2(relative_cart[:, :, 1], relative_cart[:, :, 0]).unsqueeze(2)), dim=2)
        # subtract the angle offsets (so that the angles are in accordance with pedestrian's gaze direction)
        relative[:, :, 1] -= angle_offsets.unsqueeze(1).repeat(1, obs.size(0))
        # Deleting Diagonal (to not consider a pedestrian with respect to itself)
        # [num_peds, num_peds, 2] --> [num_peds, num_peds-1, 2]
        relative = relative[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1, 2)
        # normalize angle values in [-pi, pi) interval
        relative[:, :, 1] = (relative[:, :, 1] + np.pi) % (2 * np.pi) - np.pi
        if other_values is None:  # In case of 'occupancy' pooling
            other_values = torch.ones(num_peds, num_peds - 1, self.pooling_dim, device=obs.device)
        # force use of standard occupancy to facilitate some operations
        other_values_o = torch.ones(num_peds, num_peds - 1, 1, device=obs.device) if self.include_occ else None
        oij = torch.zeros_like(relative)
        if not hasattr(self, 'shape_values') or self.shape_values.all_radius is None or \
                self.shape_values.all_angles is None:
            all_radius = torch.ones_like(oij[:, :, 0]) * self.arc_radius
            all_angles = torch.ones_like(oij[:, :, 0]) * self.arc_angle
        else:
            # use existing supplied values for an exterior configuration (most likely from ShapeConfigLSTM)
            all_radius = self.shape_values.all_radius.unsqueeze(1).repeat(1, num_peds - 1)
            all_angles = self.shape_values.all_angles.unsqueeze(1).repeat(1, num_peds - 1)
        oij[:, :, 0] = relative[:, :, 0] / all_radius * self.n_r
        oij[:, :, 1] = self.n_a / 2.0 * (relative[:, :, 1] / (all_angles / 2) + 1)
        # if range_no_violations has 0 value - index is good (neighbour in a specific cell)
        range_no_violations = torch.zeros_like(oij)
        range_no_violations[:, :, 0] = (oij[:, :, 0] < 0) + (oij[:, :, 0] >= self.n_r)
        range_no_violations[:, :, 1] = (oij[:, :, 1] < 0) + (oij[:, :, 1] >= self.n_a)
        range_mask = torch.sum(range_no_violations, dim=2) == 0
        range_mask_full = range_mask
        # range_mask == False -> outside of shape - will not count for pooling
        oij[~range_mask_full] = 0
        other_values[~range_mask_full] = 0
        if self.include_occ:
            other_values_o[~range_mask_full] = 0
        # other_values[~range_mask] = self.constant # self.constant = 0 by default; not implemented here
        oij = oij.long()
        # Flatten - numbering goes across the same radius, and going counter-clockwise (neg to pos) in angle
        oi = oij[:, :, 0] * self.n_a + oij[:, :, 1]
        # faster occupancy
        occ = torch.zeros(num_peds, self.n_a * self.n_r, self.pooling_dim, device=obs.device)
        occ_o = torch.zeros(num_peds, self.n_a * self.n_r, 1, device=obs.device) if self.include_occ else None
        # occ = self.constant*torch.ones(num_tracks, self.n**2 * self.pool_size**2, self.pooling_dim, device=obs.device)
        # Fill occupancy map with attributes - sum for elements in the same cell
        for i in range(num_peds):
            # use index_add_ so that values of multiple agents can be accumulated on the same cell
            # this requires the indexes to be one-dimensional, hence the above for loop
            occ[i].index_add_(0, oi[i], other_values[i])
            if self.include_occ:
                occ_o[i].index_add_(0, oi[i], other_values_o[i])
        if self.normalize:
            occ /= torch.clamp(torch.sum(occ, dim=1), min=1).unsqueeze(1).repeat(1, occ.shape[1], 1)
        if self.include_occ:
            occ_o /= torch.clamp(torch.sum(occ_o, dim=1), min=1).unsqueeze(1).repeat(1, occ_o.shape[1], 1)
        occ, occ_o = torch.transpose(occ, 1, 2), torch.transpose(occ_o, 1, 2) if self.include_occ else None
        occ_2d, occ_o_2d = occ.view(num_peds, -1, self.n_r, self.n_a), \
                           occ_o.view(num_peds, -1, self.n_r, self.n_a) if self.include_occ else None
        occ_summed = torch.nn.functional.lp_pool2d(occ_2d, 1, self.pool_size)
        return occ_summed

    def __init_input_dim__(self, input_dim):
        if input_dim is None:
            if self.arc:
                input_dim = self.n_r * self.n_a * self.pooling_dim
            else:
                input_dim = self.n * self.n * self.pooling_dim
        return input_dim

    ## Architectures of Encoding Grid
    def one_layer(self, input_dim=None):
        input_dim = self.__init_input_dim__(input_dim)
        return  torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.out_dim),
            torch.nn.ReLU(),)

    ## Default Layer Dims: 1024
    def two_layer(self, input_dim=None, layer_dims=None):
        input_dim = self.__init_input_dim__(input_dim)
        return  torch.nn.Sequential(
            torch.nn.Linear(input_dim, layer_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_dims[0], self.out_dim),
            torch.nn.ReLU(),)

    ## Default Layer Dims: 1024, 512
    def three_layer(self, input_dim=None, layer_dims=None):
        input_dim = self.__init_input_dim__(input_dim)
        return  torch.nn.Sequential(
            torch.nn.Linear(input_dim, layer_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_dims[0], layer_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_dims[1], self.out_dim),
            torch.nn.ReLU(),)

    def lstm_layer(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(self.out_dim, self.hidden_dim)
        self.hidden2pool = torch.nn.Linear(self.hidden_dim, self.out_dim)
        return torch.nn.Sequential(
                         torch.nn.Linear(self.n * self.n * self.pooling_dim, self.out_dim),
                         torch.nn.ReLU(),)

    def reset(self, num_tracks, device):
        self.track_mask = None
        if self.embedding_arch == 'lstm_layer':
            self.hidden_cell_state = (
                [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
                [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
            )

    def lstm_forward(self, grid):
        """ Forward process for LSTM-based grid encoding"""
        grid_embedding = self.embedding(grid)

        num_tracks = grid.size(0)
        ## If only primary pedestrian of the scene present
        if torch.sum(self.track_mask).item() == 1:
            return torch.zeros(num_tracks, self.out_dim, device=grid.device)

        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(self.track_mask, self.hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(self.track_mask, self.hidden_cell_state[1]) if m], dim=0),
        ]

        ## Update interaction-encoder LSTM
        hidden_cell_stacked = self.pool_lstm(grid_embedding, hidden_cell_stacked)
        interaction_vector = self.hidden2pool(hidden_cell_stacked[0])

        ## Save hidden-cell-states
        mask_index = [i for i, m in enumerate(self.track_mask) if m]
        for i, h, c in zip(mask_index,
                           hidden_cell_stacked[0],
                           hidden_cell_stacked[1]):
            self.hidden_cell_state[0][i] = h
            self.hidden_cell_state[1][i] = c

        return interaction_vector

    def make_grid(self, obs):
        """ Make the grids for all time-steps together 
            Only supports Occupancy and Directional pooling
        """
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
        timesteps = obs.size(0)

        grid = []
        for i in range(1, timesteps):
            obs1 = obs[i-1]
            obs2 = obs[i]
            ## Remove NANs
            track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
            obs1, obs2 = obs1[track_mask], obs2[track_mask]
            if self.type_ == 'occupancy':
                grid.append(self.occupancies(obs1, obs2))
            elif self.type_ == 'directional':
                grid.append(self.directional(obs1, obs2))
        return grid
