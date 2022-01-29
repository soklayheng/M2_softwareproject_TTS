# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import json
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

from hifigan.models import Generator as HiFiGAN
from hifigan.env import AttrDict


HIFIGAN_CONFIG = os.path.abspath('./Grad-TTS/checkpts/hifigan-config.json')
HIFIGAN_CHECKPT = os.path.abspath('./Grad-TTS/checkpts/hifigan.pt')
GRAD_CHECKPT = os.path.abspath('./Grad-TTS/checkpts/grad-tts-old.pt')
# HIFIGAN_CONFIG = os.path.abspath('./src/Grad-TTS/checkpts/hifigan-config.json')
# HIFIGAN_CHECKPT = os.path.abspath('./src/Grad-TTS/checkpts/hifigan.pt')
# GRAD_CHECKPT = os.path.abspath('./src/Grad-TTS/checkpts/grad-tts-old.pt')
TIMESTEPS = 10
SPEAKER = None

generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim, params.n_enc_channels, params.filter_channels, params.filter_channels_dp, params.n_heads,
                    params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size, params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
generator.load_state_dict(torch.load(GRAD_CHECKPT, map_location=lambda loc, storage: loc))
_ = generator.eval()

vocoder = HiFiGAN(AttrDict(json.load(open(HIFIGAN_CONFIG))))
vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
_ = vocoder.eval()
vocoder.remove_weight_norm()

cmu = cmudict.CMUDict('./Grad-TTS/resources/cmu_dictionary')


def say(sent: str):
    fout_path = os.path.abspath('./Grad-TTS/out/sample.wav')

    with torch.no_grad():
        x = torch.LongTensor(intersperse(text_to_sequence(sent, dictionary=cmu), len(symbols)))[None]
        x_lengths = torch.LongTensor([x.shape[-1]])
        _, y_dec, _ = generator.forward(x, x_lengths, n_timesteps=TIMESTEPS, temperature=1.5, stoc=False, spk=SPEAKER, length_scale=0.91)

        audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

        write(fout_path, 22050, audio)

    return fout_path
