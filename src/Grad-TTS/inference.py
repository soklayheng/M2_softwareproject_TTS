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

from pydub import AudioSegment


HIFIGAN_CONFIG = os.path.abspath('./Grad-TTS/checkpts/hifigan-config.json')
HIFIGAN_CHECKPT = os.path.abspath('./Grad-TTS/checkpts/hifigan.pt')
GRAD_CHECKPT = os.path.abspath('./Grad-TTS/checkpts/grad-tts-old.pt')

HIFIGAN_CONFIG_FR = os.path.abspath('./Grad-TTS/checkpts/hifigan-config-fr-fr.json')
HIFIGAN_CHECKPT_FR = os.path.abspath('./Grad-TTS/checkpts/hifigan-fr-fr.pt')
GRAD_CHECKPT_FR = os.path.abspath('./Grad-TTS/checkpts/grad_13.pt')
TIMESTEPS = 10
SPEAKER = None

generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim, params.n_enc_channels, params.filter_channels, params.filter_channels_dp, params.n_heads,
                    params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size, params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
generator.load_state_dict(torch.load(GRAD_CHECKPT, map_location=lambda loc, storage: loc))
_ = generator.eval()

generator_fr = GradTTS(len(symbols), params.n_spks, params.spk_emb_dim, params.n_enc_channels, params.filter_channels, params.filter_channels_dp, params.n_heads,
                    params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size, params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
generator_fr.load_state_dict(torch.load(GRAD_CHECKPT_FR, map_location=lambda loc, storage: loc))
_ = generator_fr.eval()

vocoder = HiFiGAN(AttrDict(json.load(open(HIFIGAN_CONFIG))))
vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
_ = vocoder.eval()
vocoder.remove_weight_norm()

vocoder_fr = HiFiGAN(AttrDict(json.load(open(HIFIGAN_CONFIG_FR))))
vocoder_fr.load_state_dict(torch.load(HIFIGAN_CHECKPT_FR, map_location=lambda loc, storage: loc)['generator'])
_ = vocoder_fr.eval()
vocoder_fr.remove_weight_norm()

cmu = cmudict.CMUDict('./Grad-TTS/resources/cmu_dictionary')
cmu_fr = cmudict.CMUDict('./Grad-TTS/resources/cmu_dictionary_fr')


def say(sent: str, lang: str) -> str:
    fout_path_wav = os.path.abspath('./Grad-TTS/out/sample.wav')
    fout_path_mp3 = os.path.abspath('./Grad-TTS/out/sample.mp3')

    with torch.no_grad():
        if lang == "EN":
            x = torch.LongTensor(intersperse(text_to_sequence(sent, dictionary=cmu), len(symbols)))[None]
        else:
            x = torch.LongTensor(intersperse(text_to_sequence(sent, dictionary=cmu_fr), len(symbols)))[None]
        x_lengths = torch.LongTensor([x.shape[-1]])
        _, y_dec, _ = generator.forward(x, x_lengths, n_timesteps=TIMESTEPS, temperature=1.5, stoc=False, spk=SPEAKER, length_scale=0.91)

        if lang == "EN":
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
        else:
            audio = (vocoder_fr.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

        write(fout_path_wav, 22050, audio)

        AudioSegment.from_wav(fout_path_wav).export(fout_path_mp3, format="mp3")

    return fout_path_mp3
