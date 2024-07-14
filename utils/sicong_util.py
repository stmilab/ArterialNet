"""
@author: Sicong Huang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import smtplib, ssl
import os
import mat73
import gc
import json
import time


def email_func(recipient="recipient email", sender="sender email", message="content "):
    port = 465  # For SSL
    password = "your password"
    # Create a secure SSL context
    context = ssl.create_default_context()
    message = message + "\n"
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, "Automatic Email\n" + message)
    return message


def randomize_index(input_length=100, seed=27):
    np.random.seed(seed)
    index_list = np.arange(input_length)
    np.random.shuffle(index_list)
    return index_list


def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x - mx, y - my
    r_num = np.sum(np.multiply(xm, ym))
    r_den = np.sqrt(np.multiply(np.sum(np.square(xm)), np.sum(np.square(ym))))
    r = r_num / r_den

    r = np.maximum(np.minimum(r, 1.0), -1.0)
    return r


def pad_data(data, scaler, series_len, num_features):
    data_padded = np.zeros((0, series_len, num_features))
    for i in range(len(data)):
        l = data[i].shape[0]
        data_padded = np.concatenate(
            (
                data_padded,
                np.pad(
                    scaler.transform(data[i]),
                    ((series_len - l, 0), (0, 0)),
                    "constant",
                    constant_values=-3,
                ).reshape(1, series_len, num_features),
            ),
            axis=0,
        )
    return data_padded


def array_remap(index_array):
    consecutive_attr_array = [1]
    for i in range(1, len(index_array)):
        if index_array[i] - index_array[i - 1] > 1:
            consecutive_attr_array.append(0)
        else:
            consecutive_attr_array.append(1)
    return index_array, consecutive_attr_array


def calc_consecutive(index_array, min_sequence=5, max_tolerance=0):
    index_array, c_array = array_remap(index_array)
    consecutive_dict = {}
    consecutive_dict["pairs"] = []
    consecutive_dict["original_array"] = index_array
    consecutive_dict["maximum_length"] = [0, 0, 0]
    pair = []
    gap_count = 0
    for i, e in enumerate(c_array):
        if len(pair) == 0:
            if e == 1:
                pair.append(i)
        if len(pair) == 1:
            if e == 0:
                if gap_count >= max_tolerance:
                    pair.append(i)
                else:
                    gap_count += 1
        if len(pair) == 2:
            if pair[1] - pair[0] >= min_sequence:
                consecutive_dict["pairs"].append([pair[0], pair[1], pair[1] - pair[0]])
                if consecutive_dict["maximum_length"][-1] < (pair[1] - pair[0]):
                    consecutive_dict["maximum_length"] = [
                        pair[0],
                        pair[1],
                        pair[1] - pair[0],
                    ]
            pair = []
            gap_count = 0
    return consecutive_dict
