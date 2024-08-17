"""
@author: Gabriel Mittag, TU-Berlin
"""

import os

import torch

from nisqa.lib import predict_dim, SpeechHelper, NisqaDim


class NisqaModel:
    def __init__(self, args):
        self.args = args

        self.load_model()

        self.speech_helper = SpeechHelper(
            seg_length=self.args["ms_seg_length"],
            max_length=self.args["ms_max_segments"],
            seg_hop_length=self.args["ms_seg_hop_length"],
            ms_n_fft=self.args["ms_n_fft"],
            ms_hop_length=self.args["ms_hop_length"],
            ms_win_length=self.args["ms_win_length"],
            ms_n_mels=self.args["ms_n_mels"],
            ms_sr=self.args["ms_sr"],
            ms_fmax=self.args["ms_fmax"],
            ms_channel=self.args["ms_channel"],
        )

    def predict(self, filename):
        scores = predict_dim(
            self.model,
            self.device,
            self.speech_helper,
            filename,
        )

        return scores

    def load_model(self):
        self.device = torch.device("cpu")

        if "run_device" in self.args:
            if self.args["run_device"] != "cpu":
                self.device = torch.device(self.args["run_device"])

        if self.args["pretrained_model"]:
            if os.path.isabs(self.args["pretrained_model"]):
                model_path = os.path.join(self.args["pretrained_model"])
            else:
                model_path = os.path.join(os.getcwd(), self.args["pretrained_model"])

            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=True
            )

            checkpoint["args"].update(self.args)
            self.args = checkpoint["args"]

        self.args["dim"] = True
        self.args["csv_mos_train"] = None
        self.args["csv_mos_val"] = None
        self.args["double_ended"] = False
        self.args["csv_ref"] = None

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

        self.model = NisqaDim(**self.model_args)

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

        self.model.to(self.device)
        self.model.eval()
