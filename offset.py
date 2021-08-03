import os
from os.path import join, dirname
import time
import pdb
import math
import pickle
import argparse
import subprocess
from glob import glob
from shutil import rmtree

import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
import python_speech_features

from syncnet import SyncNet


class GetOffset:
    def __init__(self, 
            video_path,
            tmp_dir='./tmp/sync', 
            reference='offset', 
            batch_size=20, vshift=10, num_layers_in_fc_layers=1024):

        self.video_path = video_path
        self.video_folder_path = dirname(self.video_path)
        self.tmp_dir = tmp_dir
        self.reference = reference
        self.device = 'cuda'
        self.batch_size = batch_size
        self.vshift = vshift
        self.weight_path = './tmp/model_weight/syncnet_v2.model'

        if os.path.exists(os.path.join(self.tmp_dir, self.reference)):
            rmtree(join(self.tmp_dir, self.reference))
        os.makedirs(join(self.tmp_dir, self.reference))

        self.model = SyncNet(num_layers_in_fc_layers=num_layers_in_fc_layers).to(self.device)
        
        state_dict = torch.load(self.weight_path, map_location=self.device)
        model_state = self.model.state_dict()
        for name, param in state_dict.items():
            model_state[name].copy_(param)
		
        self.model.eval()

    def extract_data(self):
        command = ("ffmpeg -loglevel error -y -i {0} -threads 1 -f image2 {1}".format(self.video_path, join(self.tmp_dir, self.reference, '%06d.jpg')))
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel error -y -i {0} -ac 1 -vn -acodec pcm_s16le -ar 16000 {1}".format(self.video_path, join(self.tmp_dir, self.reference, 'audio.wav')))
        output = subprocess.call(command, shell=True, stdout=None)

    def prep_data(self):
        # Get video frames
        images = []
        flist = glob(join(self.tmp_dir, self.reference, '*.jpg'))
        flist.sort()
        for fname in flist:
            images.append(cv2.resize(cv2.imread(fname), (224, 224)))
		
        im = np.stack(images, axis=3)
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.from_numpy(im.astype(float)).float()

        # Load audio
        sample_rate, audio = wavfile.read(join(self.tmp_dir, self.reference, 'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])

        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.from_numpy(cc.astype(float)).float()

        min_length = min(len(images), math.floor(len(audio)/640))
		
        return imtv, cct, min_length

    def calc_pdist(self, feat1, feat2, vshift=10):
        win_size = vshift * 2 + 1
        feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
        dists = []

        for i in range(0, len(feat1)):
            dists.append(torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i+win_size, :]))

        return dists

    def evaluate(self):
        imtv, cct, min_length = self.prep_data()
        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        for i in range(0, lastframe, self.batch_size):
            im_batch = [imtv[:, :, vframe:vframe+5, :, :] for vframe in range(i, min(lastframe, i+self.batch_size))]
            im_in = torch.cat(im_batch, 0) # (20, 3, 5, 224, 224)
            im_out = self.model.forward_lip(im_in.to(self.device))
            im_feat.append(im_out.data.cpu())

            cc_batch = [cct[:, :, :, vframe*4: vframe*4+20] for vframe in range(i, min(lastframe, i+self.batch_size))]
            cc_in = torch.cat(cc_batch, 0)
            cc_out = self.model.forward_audio(cc_in.to(self.device))
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        dists = self.calc_pdist(im_feat, cc_feat, vshift=self.vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)

        minval, minidx = torch.min(mdist, 0)

        offset = self.vshift - minidx # unit of offset is frame, one for 0.04 seconds
        conf = torch.median(mdist) - minval

        return offset, conf

