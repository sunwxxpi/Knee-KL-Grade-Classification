import numpy as np

import torch
from torch import nn

class EarlyStopping:
    def __init__(self, args, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
        """
        self.args = args
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def save_checkpoint(self, val_loss, model, args, fold, epoch):
        if self.verbose:
            print(f"Valid Loss ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving Model ...")
            
        image_size_tuple = (args.image_size, args.image_size)
        
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), f'./models/{args.model_type}/{image_size_tuple}/{fold}fold_epoch{epoch}.pt')
        else:
            torch.save(model.state_dict(), f'./models/{args.model_type}/{image_size_tuple}/{fold}fold_epoch{epoch}.pt')
        # torch.save(model, f"./models/{args.model_type}/{image_size_tuple}/{fold}fold_epoch{epoch}.pt")
            
        self.val_loss_min = val_loss
        
    def __call__(self, val_loss, model, args, fold, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args, fold, epoch)
        elif self.best_score - self.delta > score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.best_score < score:
                self.best_score = score
            self.save_checkpoint(val_loss, model, args, fold, epoch)
            self.counter = 0