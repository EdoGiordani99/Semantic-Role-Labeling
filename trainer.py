import os
import torch
import datetime

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from evaluation import compute_f1


class Trainer():

    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 epochs: int,
                 device: str,
                 config: dict,
                 compute_f1: bool = False,
                 save_folder: str = 'store',
                 save_history: bool = True,
                 save_model: bool = True,
                 save_best: bool = True):
        """
        This class trains your model!
        Args:
            model (nn.Module): the model to train
            optimizer (torch.Optim): optimizer to use
            train_dataloader (Dataloader): train data loadert
            valid_dataloader (Dataloader): valid data loadert
            epochs (int): epochs to train
            device (str): 'cuda' or 'cpu'
            config (dict): configuration dictionary with all hyperparameters
            compute_f1 (bool): if true, f1 score is computed during training
            save_folder (str): path where you want the logs to be saved
            save_history (bool): if true, history is saved in save_folder
            save_model (bool): if true, model is saved in save_folder
            save_best (bool): if true, model is saved only when it reaches a lower
              validation loss
        Returns:
            history (dict): dictionary containing 'train_epoch_loss',
              'train_step_loss', 'train_f1', 'valid_epoch_loss', 'valid_step_loss',
              'valid_f1'
        """

        self.model = model
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.epochs = epochs
        self.device = device
        self.config = config

        self.compute_f1 = compute_f1

        self.save_folder = save_folder
        self.save_history = save_history
        self.save_best = save_best
        self.save_model = save_model

    def train(self):

        history = {'train_step_loss': [],
                   'train_epoch_loss': [],
                   'train_f1': [],
                   'valid_step_loss': [],
                   'valid_epoch_loss': [],
                   'valid_f1': []}

        t = datetime.datetime.now()
        daytime = str(t.year) + ':' + str(t.month) + ':' + str(t.day) + '_' + str(t.hour) + ':' + str(t.minute)

        # where model will be saved
        if not os.path.exists('./store'):
            os.makedirs('./store')

        save_path = self.save_folder + '/' + self.config['model_name'] + '_' + daytime
        os.mkdir(save_path)

        # Config paramenters
        torch.save(self.config, save_path + '/config.pth')

        best_valid_loss = 10e10

        print('Training...')

        for i in range(1, self.epochs + 1):

            if i < 10:
                print(f'\nEpoch 0{i}:')
            if i >= 10:
                print(f'\nEpoch {i}:')

            epoch_train_f1 = 0
            epoch_valid_f1 = 0

            # TRAINING STEP
            t_progbar = tqdm(enumerate(self.train_dataloader),
                             total=self.train_dataloader.__len__(),
                             desc='Training: ')
            train_loss = 0
            train_true_all = []
            train_pred_all = []

            for i, train_batch in t_progbar:

                self.optimizer.zero_grad()
                t_batch = {k: v.to(self.device) for k, v in train_batch.items()}

                if self.compute_f1:
                    outputs = self.model(**t_batch, compute_predictions=True)
                else:
                    outputs = self.model(**t_batch)

                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()

                t_loss = round(loss.item(), 4)

                history['train_step_loss'].append(t_loss)
                train_loss += t_loss

                if self.compute_f1:
                    yt_true = train_batch['labels']
                    yt_pred = outputs['preds']

                    train_true_all += (yt_true.tolist())
                    train_pred_all += (yt_pred.tolist())

                    t_f1 = round(compute_f1(yt_true, yt_pred), 4)
                    epoch_train_f1 += t_f1

                else:
                    t_f1 = 0

                t_progbar.set_postfix({'loss': t_loss, 'f1': t_f1})

            epoch_train_loss = round(train_loss / self.train_dataloader.__len__(), 4)
            history['train_epoch_loss'].append(epoch_train_loss)

            epoch_train_f1 = compute_f1(torch.Tensor(train_true_all),
                                        torch.Tensor(train_pred_all))
            history['train_f1'].append(epoch_train_f1)

            # VALIDATION STEP
            v_progbar = tqdm(enumerate(self.valid_dataloader),
                             total=self.valid_dataloader.__len__(),
                             desc='Validation: ')
            valid_loss = 0

            for i, valid_batch in v_progbar:

                v_batch = {k: v.to(self.device) for k, v in valid_batch.items()}

                if self.compute_f1:
                    outputs = self.model(**v_batch, compute_predictions=True)
                else:
                    outputs = self.model(**v_batch)

                v_loss = round(outputs['loss'].item(), 4)
                history['valid_step_loss'].append(v_loss)

                valid_loss += v_loss

                valid_pred_all = []
                valid_true_all = []

                if self.compute_f1:
                    yv_true = valid_batch['labels']
                    yv_pred = outputs['preds']

                    valid_true_all += (yv_true.tolist())
                    valid_pred_all += (yv_pred.tolist())

                    v_f1 = round(compute_f1(yv_true, yv_pred), 4)
                    epoch_valid_f1 += v_f1

                else:
                    v_f1 = 0

                v_progbar.set_postfix({'val loss': v_loss, 'val f1': v_f1})

            epoch_valid_loss = round(valid_loss / self.valid_dataloader.__len__(), 4)
            history['valid_epoch_loss'].append(epoch_valid_loss)

            epoch_valid_f1 = compute_f1(torch.Tensor(valid_true_all),
                                        torch.Tensor(valid_pred_all))
            history['valid_f1'].append(epoch_valid_f1)

            print(
                f'\nTrain Loss: {epoch_train_loss}  Train F1: {epoch_train_f1}  Valid Loss: {epoch_valid_loss}  Valid F1: {epoch_valid_f1}')

            # SAVING MODEL AND HISTORY
            if self.save_model:
                if self.save_best:
                    if epoch_valid_loss < best_valid_loss:
                        name = save_path + '/model.pth'
                        torch.save(self.model.state_dict(), name)
                else:
                    model_name = save_path + '/model.pth'
                    torch.save(self.model.state_dict(), model_name)

            if self.save_history:
                history_name = save_path + '/history.pth'
                torch.save(history, history_name)

        return history, save_path
