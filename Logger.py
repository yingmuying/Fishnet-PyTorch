import os
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from cycler import cycler
import numpy as np

import seaborn as sn
import pandas as pd

# x axis of plot 
LOG_KEYS = {
    "train":"epoch",
    "valid":"epoch",
}

# y axis of plot
LOG_VALUES = {
    "train":["loss_val", "loss_avg", "top1_val", "top1_avg", "top5_val", "top5_avg"],
    "valid":["loss_val", "loss_avg", "top1_val", "top1_avg", "top5_val", "top5_avg"]
}

class Logger:
      
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_file = save_dir + "/log.txt"
        self.buffers = []

    def will_write(self, line):
        print(line)
        self.buffers.append(line)

    def flush(self):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(self.buffers))
            f.write("\n")
        self.buffers = []

    def write(self, line):
        self.will_write(line)
        self.flush()

    def log_write(self, learn_type, **values):
        """log write in buffers

        ex ) log_write("train", epoch=1, loss=0.3)

        Parmeters:
            learn_type : it must be train, valid or test
            values : values keys in LOG_VALUES
        """
        for k in values.keys():
            if k not in LOG_VALUES[learn_type] and k != LOG_KEYS[learn_type]:
                raise KeyError("%s Log %s keys not in log"%(learn_type, k))
        log = "[%s] %s"%(learn_type, json.dumps(values))
        self.will_write(log)
        if learn_type != "train":
            self.flush()
        
    def log_parse(self, log_key):
        log_dict = OrderedDict()
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line) == 1 or not line.startswith("[%s]"%(log_key)): 
                    continue
                # line : ~~
                line = line[line.find("] ") + 2:] # ~~
                line_log = json.loads(line)

                train_log_key = line_log[LOG_KEYS[log_key]]
                line_log.pop(LOG_KEYS[log_key], None)
                log_dict[train_log_key] = line_log

        return log_dict
    
    def log_plot(self, log_key, 
                 figsize=(12, 12), title="plot", colors=["C1", "C2"]):
        """Plotting Log graph

        If mode is jupyter then call plt.show.
        Or, mode is slack then save image and return save path

        Parameters:
            log_key : train, valid, test
            mode : jupyter or slack
            figsize : argument of plt
            title : plot title
        """
        fig = plt.figure(figsize=figsize)
        plt.title(title)
        plt.legend(LOG_VALUES[log_key], loc="best")
        
        ax = plt.subplot(111)
        colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(LOG_VALUES[log_key])))
        ax.set_prop_cycle(cycler('color', colors))
        
        log_dict = self.log_parse(log_key)
        x = log_dict.keys()
        for keys in LOG_VALUES[log_key]:
            if keys not in list(log_dict.values())[0]:
                continue
            y = [v[keys] for v in log_dict.values()]
            
            label = keys + ", max : %f"%(max(y))
            ax.plot(x, y, marker="o", linestyle="solid", label=label)
            if max(y) > 1:
                ax.set_ylim([min(y)-1, y[0]+1])
        ax.legend(fontsize=30)

        plt.show()
        
    def report(self):
        for k, v in self.log_parse("test").items():
            print(k, v)
        
        self.log_plot("train", title="Train Log Graph", figsize=(36, 12))
        self.log_plot("valid", title="Valid Acc Graph", figsize=(36, 12))
        
        labels = None
        with open(self.log_file, "r", encoding="utf-8") as f:
            for z in f.readlines():
                if z in "[Test Labels]":
                    labels = z.strip().split(":")[1].split(",")
        
        test_confusion = np.load(self.save_dir + "/test_confusion.npy")
        vmin, vmax = test_confusion.min(), test_confusion.max()
        labels = labels if labels is not None else ["X" for _ in range(len(test_confusion))]
        df_cm = pd.DataFrame(test_confusion, 
                     index   = labels,
                     columns = labels)
        
        plt.figure()
        sn.heatmap(df_cm, annot=True, vmin=vmin, vmax=vmax, fmt="d")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.show()
