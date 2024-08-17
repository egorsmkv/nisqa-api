"""
@author: Gabriel Mittag, TU-Berlin
"""

import os

import torch
import pandas as pd

from nisqa.lib import predict_dim, SpeechQualityDataset, NISQA_DIM


class NisqaModel:
    """
    nisqaModel: Main class that loads the model and the datasets. Contains
    the training loop, prediction, and evaluation function.
    """

    def __init__(self, args):
        self.args = args

        self._load_model()
        self._load_datasets()

    def predict(self):
        _, _, scores = predict_dim(
            self.model,
            self.ds_val,
            self.args["tr_bs_val"],
            self.dev,
            num_workers=self.args["tr_num_workers"],
        )

        return scores

    def _load_datasets(self):
        data_dir = os.path.dirname(self.args["filename"])
        file_name = os.path.basename(self.args["filename"])
        df_val = pd.DataFrame([file_name], columns=["filename"])

        # creating Datasets
        self.ds_val = SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir=data_dir,
            filename_column="filename",
            mos_column="predict_only",
            seg_length=self.args["ms_seg_length"],
            max_length=self.args["ms_max_segments"],
            seg_hop_length=self.args["ms_seg_hop_length"],
            transform=None,
            ms_n_fft=self.args["ms_n_fft"],
            ms_hop_length=self.args["ms_hop_length"],
            ms_win_length=self.args["ms_win_length"],
            ms_n_mels=self.args["ms_n_mels"],
            ms_sr=self.args["ms_sr"],
            ms_fmax=self.args["ms_fmax"],
            ms_channel=self.args["ms_channel"],
            double_ended=self.args["double_ended"],
            dim=self.args["dim"],
            filename_column_ref=None,
        )

    def _load_model(self):
        self.dev = torch.device("cpu")

        if "run_device" in self.args:
            if self.args["run_device"] != "cpu":
                self.dev = torch.device(self.args["run_device"])

        # if True overwrite input arguments from pretrained model
        if self.args["pretrained_model"]:
            if os.path.isabs(self.args["pretrained_model"]):
                model_path = os.path.join(self.args["pretrained_model"])
            else:
                model_path = os.path.join(os.getcwd(), self.args["pretrained_model"])

            checkpoint = torch.load(
                model_path, map_location=self.dev, weights_only=True
            )

            # update checkpoint arguments with new arguments
            checkpoint["args"].update(self.args)
            self.args = checkpoint["args"]

        self.args["dim"] = True
        self.args["csv_mos_train"] = None  # column names hardcoded for dim models
        self.args["csv_mos_val"] = None
        self.args["double_ended"] = False
        self.args["csv_ref"] = None

        # Load Model
        self.model_args = {
            "ms_seg_length": self.args["ms_seg_length"],
            "ms_n_mels": self.args["ms_n_mels"],
            "cnn_model": self.args["cnn_model"],
            "cnn_c_out_1": self.args["cnn_c_out_1"],
            "cnn_c_out_2": self.args["cnn_c_out_2"],
            "cnn_c_out_3": self.args["cnn_c_out_3"],
            "cnn_kernel_size": self.args["cnn_kernel_size"],
            "cnn_dropout": self.args["cnn_dropout"],
            "cnn_pool_1": self.args["cnn_pool_1"],
            "cnn_pool_2": self.args["cnn_pool_2"],
            "cnn_pool_3": self.args["cnn_pool_3"],
            "cnn_fc_out_h": self.args["cnn_fc_out_h"],
            "td": self.args["td"],
            "td_sa_d_model": self.args["td_sa_d_model"],
            "td_sa_nhead": self.args["td_sa_nhead"],
            "td_sa_pos_enc": self.args["td_sa_pos_enc"],
            "td_sa_num_layers": self.args["td_sa_num_layers"],
            "td_sa_h": self.args["td_sa_h"],
            "td_sa_dropout": self.args["td_sa_dropout"],
            "td_lstm_h": self.args["td_lstm_h"],
            "td_lstm_num_layers": self.args["td_lstm_num_layers"],
            "td_lstm_dropout": self.args["td_lstm_dropout"],
            "td_lstm_bidirectional": self.args["td_lstm_bidirectional"],
            "td_2": self.args["td_2"],
            "td_2_sa_d_model": self.args["td_2_sa_d_model"],
            "td_2_sa_nhead": self.args["td_2_sa_nhead"],
            "td_2_sa_pos_enc": self.args["td_2_sa_pos_enc"],
            "td_2_sa_num_layers": self.args["td_2_sa_num_layers"],
            "td_2_sa_h": self.args["td_2_sa_h"],
            "td_2_sa_dropout": self.args["td_2_sa_dropout"],
            "td_2_lstm_h": self.args["td_2_lstm_h"],
            "td_2_lstm_num_layers": self.args["td_2_lstm_num_layers"],
            "td_2_lstm_dropout": self.args["td_2_lstm_dropout"],
            "td_2_lstm_bidirectional": self.args["td_2_lstm_bidirectional"],
            "pool": self.args["pool"],
            "pool_att_h": self.args["pool_att_h"],
            "pool_att_dropout": self.args["pool_att_dropout"],
        }

        if self.args["double_ended"]:
            self.model_args.update(
                {
                    "de_align": self.args["de_align"],
                    "de_align_apply": self.args["de_align_apply"],
                    "de_fuse_dim": self.args["de_fuse_dim"],
                    "de_fuse": self.args["de_fuse"],
                }
            )

        self.model = NISQA_DIM(**self.model_args)

        # Load weights if pretrained model is used
        if self.args["pretrained_model"]:
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint["model_state_dict"], strict=True
            )

            if missing_keys:
                print("missing_keys:")
                print(missing_keys)
            if unexpected_keys:
                print("unexpected_keys:")
                print(unexpected_keys)

        self.model.to(self.dev)
        self.model.eval()
