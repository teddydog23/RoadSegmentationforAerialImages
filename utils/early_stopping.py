import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode='max'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.best_model_wts = None

    def __call__(self, score, model):
        if self.mode == 'max':
            score = score
        else:
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0

    def load_best_weights(self, model):
        model.load_state_dict(self.best_model_wts)
