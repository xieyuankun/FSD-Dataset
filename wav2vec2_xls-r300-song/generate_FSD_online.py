from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
import raw_dataset as dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor,Wav2Vec2Config
import numpy as np


def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform


def normalization(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    distance = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(distance)
    return norm_data


def test_on_FSD(feat_model_path):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = torch.load(feat_model_path)
    model1.eval()

    config = Wav2Vec2Config.from_json_file("/data3/xyk/huggingface/wav2vec2-xls-r-300m/config.json")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/")
    model = Wav2Vec2Model.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/").cuda()
    model.config.output_hidden_states = True
    labeldict = {"spoof": 1, "bonafide": 0}
    with open('FC1_result.txt', 'w') as cm_score_file:
        asvspoof_raw = dataset.FSDeval()
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, labels  = asvspoof_raw[idx]
            waveform = waveform.to(device)
            waveform = pad_dataset(waveform).to('cpu')

            input_values = processor(waveform, sampling_rate=16000,
                                     return_tensors="pt").input_values.cuda()  

            with torch.no_grad():
                wav2vec2 = model(input_values).last_hidden_state.cuda()  
                # wav2vec2 = model(waveform)[0].cuda()  
            # wav2vec2 = wav2vec2.squeeze(dim=0)
            # wav2vec2 = normalization(wav2vec2)
            # wav2vec2 = wav2vec2.unsqueeze(dim=0)
            # wav2vec2 = input_values.transpose(1, 2).cuda()
            w2v2, audio_fn= wav2vec2, filename
            w2v2 = w2v2.to('cpu')
            this_feat_len = w2v2.shape[1]
            print(w2v2.shape)
            w2v2 = w2v2.unsqueeze(dim=0)
            w2v2 = w2v2.transpose(2, 3).to(device)
            # labels = torch.zeros((w2v2.shape[0]))

            feats, w2v2_outputs = model1(w2v2)
            score = F.softmax(w2v2_outputs)[:, 0]

            cm_score_file.write('%s %s %s\n' % (
            audio_fn, score.item(), "spoof" if labels== "spoof" else "bonafide"))



if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    test_on_FSD(model_path)