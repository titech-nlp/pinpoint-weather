
# -*- coding: utf-8 -*-

import os
import json
import torch
import pickle

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from lib.constant import IDs, Tokens
from lib.preprocessing import fill_bos_eos_id

class WNCGDataset(Dataset):
    def __init__(self, dataset_dir, vocab, data_split="train", max_size=None):
        self.src_gpv_list = []
        self.src_amedas_list = []
        self.src_meta_list = []
        self.tgt_label_list = []
        self.tgt_comment_list = []
        self.vocab = vocab
        self.data_split = data_split

        # load_dataset
        wncg_comments = json.load(open(os.path.join(dataset_dir, "wncg-comment.json")))
        for key, comment_data in wncg_comments.items():
            if data_split == comment_data["data_split"]:
                # filter by max_size
                if max_size is not None:
                    if max_size < len(self.tgt_comment_list):
                        break
                save_data = pickle.load(open(os.path.join(dataset_dir, key), 'rb'))
                self.src_gpv_list.append(save_data["gpv"])
                self.src_amedas_list.append(save_data["amedas"])
                self.src_meta_list.append(save_data["meta"])
                self.tgt_label_list.append(comment_data["topic"])
                self.tgt_comment_list.append(comment_data["comment"])

    def __len__(self):
        return len(self.tgt_comment_list)

    def get_weather_label(self, topics):
        sunny_label = 1 if "晴れ" in topics else 0
        cloudy_label = 1 if "曇り" in topics else 0
        rain_label = 1 if "雨" in topics else 0
        snow_label = 1 if "雪" in topics else 0
        return (sunny_label, cloudy_label, rain_label, snow_label)

    def get_raw_item(self, idx):
        src_gpv = self.src_gpv_list[idx]
        src_amedas = self.src_amedas_list[idx]
        src_meta = self.src_meta_list[idx]
        tgt_label = self.tgt_label_list[idx]
        tgt_comment = self.tgt_comment_list[idx]
        return [src_gpv, src_amedas, src_meta, tgt_label, tgt_comment]

    def __getitem__(self, idx):
        src_gpv = torch.tensor(self.src_gpv_list[idx], dtype=torch.float)
        src_amedas = torch.tensor(self.src_amedas_list[idx], dtype=torch.float)
        _meta_area = self.src_meta_list[idx]["area"]
        _meta_month = self.src_meta_list[idx]["month"]
        _meta_day = self.src_meta_list[idx]["day"]
        _meta_time = self.src_meta_list[idx]["time"]
        _meta_week = self.src_meta_list[idx]["week"]
        src_meta = torch.LongTensor([_meta_area, _meta_month, _meta_day, _meta_time, _meta_week])

        # create weather label
        tgt_sunny_label, tgt_cloudy_label, tgt_rain_label, tgt_snow_label = self.get_weather_label(self.tgt_label_list[idx])
        tgt_label = torch.LongTensor([tgt_sunny_label, tgt_cloudy_label, tgt_rain_label, tgt_snow_label])
        tgt_comment = torch.LongTensor(fill_bos_eos_id(list(map(self.vocab.stoi, self.tgt_comment_list[idx].strip().split()))))
        return src_gpv, src_amedas, src_meta, tgt_label, tgt_comment

def my_collate(batch):
    src_gpv = torch.stack([item[0] for item in batch], dim=0)
    src_amedas = torch.stack([item[1] for item in batch], dim=0)
    src_meta = torch.stack([item[2] for item in batch], dim=0)
    tgt_label = torch.stack([item[3] for item in batch], dim=0)
    tgt_comment = pad_sequence([item[4] for item in batch], batch_first=True, padding_value=IDs.PAD.value)
    return [src_gpv, src_amedas, src_meta, tgt_label, tgt_comment]
