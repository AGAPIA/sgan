import argparse
import os
import torch
import config
from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--external_test', type=bool, default=False)
parser.add_argument('--external', type=bool, default=False)

parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    #args['loader_num_workers'] = 0
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cpu() #.cuda()
    generator.eval()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cpu() for tensor in batch] #[tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        #_args['loader_num_workers'] = 0
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))

# FLASK SESSION GLOBAL DEFINES
import numpy as np
from flask import Flask, jsonify, request
import json
import io
app = Flask(__name__)

NUM_FRAMES_TO_OBSERVE = 8
NUM_FRAMES_TO_PREDICT = 8
MAIN_AGENT_NAME = "main" # The agent under evaluation
MAIN_AGENT_INDEX = 0
generator = None
history_pos = {} # History of positons for a given agent name
#----------

# This code runs the inference for one frome
# Input params:
# agentsObservedPos dict of ['agentName'] -> position as np array [2], all agents observed in this frame
# optional: forcedHistoryDict -> same as above but with NUM_FRAMES_OBSERVED o neach agent, allows you to force / set history
# Output : returns the position of the 'main' agent
def DoModelInferenceForFrame(agentsObservedOnFrame, forcedHistoryDict = None):
    global history_pos

    # Update the history if forced param is used
    if forcedHistoryDict != None:
        for key, value in forcedHistoryDict.items():
            assert isinstance(value, np.ndarray), "value is not instance of numpy array"
            assert value.shape is not (NUM_FRAMES_TO_OBSERVE, 2)
            history_pos[key] = value

    # Update the history of agents seen with the new observed values
    for key, value in agentsObservedOnFrame.items():
        # If agent was not already in the history pos, init everything with local value
        if key not in history_pos:
            history_pos[key] = np.tile(value, [NUM_FRAMES_TO_OBSERVE, 1])
        else:  # Else, just put his new pos in the end of history
            values = history_pos[key]
            values[0:NUM_FRAMES_TO_OBSERVE - 1] = values[1:NUM_FRAMES_TO_OBSERVE]
            values[NUM_FRAMES_TO_OBSERVE - 1] = value

    # Do simulation using the model
    # ------------------------------------------

    # Step 1: fill the input
    numAgentsThisFrame = len(agentsObservedOnFrame)

    # Absolute observed trajectories
    obs_traj = np.zeros(shape=(NUM_FRAMES_TO_OBSERVE, numAgentsThisFrame, 2), dtype=np.float32)
    # Zero index is main, others are following
    obs_traj[:, MAIN_AGENT_INDEX, :] = history_pos[MAIN_AGENT_NAME]
    index = 1
    indexToAgentNameMapping = {}
    indexToAgentNameMapping[MAIN_AGENT_INDEX] = MAIN_AGENT_NAME
    for key, value in agentsObservedOnFrame.items():
        if key != MAIN_AGENT_NAME:
            obs_traj[:,index,:] = history_pos[key]
            indexToAgentNameMapping[index] = key
            index += 1

    # Relative observed trajectories
    obs_traj_rel = np.zeros(shape=(NUM_FRAMES_TO_OBSERVE, numAgentsThisFrame, 2), dtype=np.float32)
    obs_traj_rel[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]

    seq_start_end = np.array([[0, numAgentsThisFrame]])  # We have only 1 batch containing all agents
    # Transform them to torch tensors
    obs_traj = torch.from_numpy(obs_traj).type(torch.float)
    obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float)
    seq_start_end = torch.from_numpy(seq_start_end)

    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    # Take the first predicted position and add it to history
    pred_traj_fake = pred_traj_fake.detach().numpy()
    newMainAgentPos = pred_traj_fake[0][0]  # Agent 0 is our main agent

    return newMainAgentPos


@app.route('/predict', methods=['POST'])
def predict():
    if request.method =='POST':
        # Get the file from request
        #file = request.files['file']
        dataReceived = json.loads(request.data)

        agentsObservedThisFrame = dataReceived['agentsPosThisFrame']
        agentsForcedHistory = dataReceived['agentsForcedHistoryPos'] if 'agentsForcedHistoryPos' in dataReceived else None

        # Read all agents data received
        #-----------------------------------------
        #agentIndex = 1
        agentsObservedPos = {}
        for key,value in agentsObservedThisFrame.items():
            value = np.array(value, dtype=np.float32)
            if key == MAIN_AGENT_NAME:
                agentsObservedPos[MAIN_AGENT_NAME] = value
            else:
                agentsObservedPos[key] = value

        forcedHistoryPos = None
        if agentsForcedHistory is not None:
            forcedHistoryPos = {}
            for key, value in agentsForcedHistory.items():
                value = np.array(value, dtype=np.float32)
                if key == MAIN_AGENT_NAME:
                    forcedHistoryPos[MAIN_AGENT_NAME] = value
                else:
                    forcedHistoryPos[key] = value

        # Then do model inference for agents observed on this frame
        # Get back the new position for main agent and return it to caller
        newMainAgentPos = DoModelInferenceForFrame(agentsObservedPos, forcedHistoryPos)
        return jsonify(newMainAgentPos = str(list(newMainAgentPos)))


def deloyModelForFlaskInference():
    checkpoint = torch.load(args.model_path)

    global generator
    generator = get_generator(checkpoint)
    _args = AttrDict(checkpoint['args'])
    #_args['loader_num_workers'] = 0

def startExternalTest():
    deloyModelForFlaskInference()

    historyMainAgent = np.zeros(shape=(NUM_FRAMES_TO_OBSERVE, 2))
    agentsObservedPos = {MAIN_AGENT_NAME : np.array([0,0], dtype=np.float32)}

    for frameIndex in range(100):
        forcedHistory = None
        if frameIndex == 0:
            forcedHistory = {MAIN_AGENT_NAME : historyMainAgent}

        newMainAgentPos = DoModelInferenceForFrame(agentsObservedPos, forcedHistory)
        print(f"Frame {frameIndex}: {newMainAgentPos}")
        agentsObservedPos[MAIN_AGENT_NAME] = newMainAgentPos

"""
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     non_linear_ped, loss_mask, seq_start_end) = batch

    ade, fde = [], []
    total_traj += pred_traj_gt.size(1)

    for _ in range(num_samples):
        pred_traj_fake_rel = generator(
            obs_traj, obs_traj_rel, seq_start_end
        )
        pred_traj_fake = relative_to_abs(
            pred_traj_fake_rel, obs_traj[-1]
        )
        ade.append(displacement_error(
            pred_traj_fake, pred_traj_gt, mode='raw'
        ))
        fde.append(final_displacement_error(
            pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
        ))
"""

if __name__ == '__main__':
    args = parser.parse_args()
    config.setDevice(args.use_gpu)


    if False and args.external == True:
        deloyModelForFlaskInference()
        app.run()
    elif False and args.external_test == True:
        startExternalTest()
    else: # normal evaluation
        main(args)



