import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import wandb
import os
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from datetime import datetime
import yaml
from CoCa.utils.plot import plot_generation_with_input
from CoCa.module.CausalConceptVAE import CausalConceptVAE

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, path='model', model_name='VAE', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.training_loss = None
        self.acc = 0
        self.delta = delta
        self.path = path
        self.model_name = model_name
        self.trace_func = trace_func

    def __call__(self, val_loss, acc, model, training_loss=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, acc, model)
            if training_loss!=None:
                self.training_loss = training_loss
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, acc, model)
            self.counter = 0
            if training_loss!=None:
                self.training_loss = training_loss


    def save_checkpoint(self, val_loss, acc, model):
        '''Saves model when validation loss decrease.'''

        self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # for name, parameters in model.named_parameters():
        #     print(f"name: {name}, mean: {torch.mean(parameters)}, std: {torch.std(parameters)}, val: {parameters[0][0]}")
        #     break

        # torch.save(model.state_dict(), os.path.join(self.path, f"{self.model_name}_val_loss_{val_loss}_matric_{acc}.pth"))
        torch.save(model.state_dict(), os.path.join(self.path, f"{self.model_name}_best.pth"))
        self.val_loss_min = val_loss
        self.acc = acc


class Trainer():
    def __init__(
        self,
        model:nn.Module,
        training_dataset: data.Dataset,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        num_workers: int,
        train_val_split: float = 0.9,
        model_save_path: str = f'model_save_path/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        ):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model

        self.batch_size = batch_size
        train_idx, val_idx = train_test_split(list(range(training_dataset.__len__())), train_size=train_val_split, shuffle=True)
        data_train = data.Subset(training_dataset, train_idx)
        data_val = data.Subset(training_dataset, val_idx)
        self.train_dataloader = data.DataLoader(data_train, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory = True, prefetch_factor=batch_size*2)
        self.val_dataloader = data.DataLoader(data_val, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory = True, prefetch_factor=batch_size*2)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        patience = 5
        self.model_save_path = model_save_path
        self.early_stopping = EarlyStopping(patience=patience, path=model_save_path, model_name = model.__class__.__name__)

        self.num_epochs = num_epochs

        parameter_dict = {
            "dataset": training_dataset.__class__.__name__,
            "model": model.__class__.__name__,
            "model_params": model.get_hyperparamters(),
            "trainer_params": {
                "batch_size": self.batch_size,
                "learning_rate": learning_rate,
                "num_epochs": self.num_epochs,
                "num_workers": num_workers,
                "early_stopping_patience": patience
            }
        }

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        with open(os.path.join(model_save_path, "training_parameters.yaml"), "w") as f:
            yaml.dump(parameter_dict, f)

    def train(self, save_samples: bool = False):
        self.model = self.model.to(self.device)
        
        for epoch in range(self.num_epochs):

            self.model.train()
            train_loss = 0
            train_reconstruct_loss=0
            train_kld_loss=0
            train_distance = 0
            with tqdm(total=len(self.train_dataloader), postfix={'epoch': epoch}) as train_bar:
                for batch_idx, (x, y) in enumerate(self.train_dataloader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.model.loss_function(*output, M_N=0.5, batch_idx=batch_idx)
                    loss['loss'].backward()
                    self.optimizer.step()
                    train_loss += loss['loss'].item()
                    train_reconstruct_loss += loss['Reconstruction_Loss'].item()
                    train_kld_loss += loss['KLD'].item()
                    train_distance += torch.mean(torch.abs(x - output[0])).item()
                    train_bar.update(1)

            self.scheduler.step()
            train_loss /= self.train_dataloader.dataset.__len__()
            train_reconstruct_loss /= self.train_dataloader.dataset.__len__()
            train_kld_loss /= self.train_dataloader.dataset.__len__()
            train_distance /= -self.train_dataloader.dataset.__len__()
            train_distance *= 255
            train_bar.set_description(f'avg_train_loss: {train_loss}; avg_train_dist: {train_distance} epoch: {epoch}')

            self.model.eval()
            val_loss = 0
            val_reconstruct_loss=0
            val_kld_loss=0
            val_distance = 0
            with tqdm(total=len(self.val_dataloader)) as val_bar:
                for batch_idx, (x, y) in enumerate(self.val_dataloader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)
                    loss = self.model.loss_function(*output, M_N=0.5, batch_idx=batch_idx)
                    val_loss += loss['loss'].item()
                    val_reconstruct_loss += loss['Reconstruction_Loss'].item()
                    val_kld_loss += loss['KLD'].item()
                    val_distance += torch.mean(torch.abs(x - output[0])).item()
                    val_bar.update(1)

                    if save_samples and batch_idx == 0:
                        save_path = os.path.join(self.early_stopping.path, "images")
                        os.makedirs(save_path, exist_ok=True)
                        plot_generation_with_input(
                            x[:3],
                            output[0][:3],
                            save_path=os.path.join(save_path, f"epoch_{epoch}.png")
                        )

            val_loss /= self.val_dataloader.dataset.__len__()
            val_reconstruct_loss /= self.val_dataloader.dataset.__len__()
            val_kld_loss /= self.val_dataloader.dataset.__len__()
            val_distance /= -self.val_dataloader.dataset.__len__()
            val_distance *= 255
            val_bar.set_description(f'avg_val_loss: {val_loss}; avg_val_dist: {val_distance}; epoch: {epoch}.')
            wandb.log({
                    'train_loss': train_loss, 
                    'train_reconstruction_Loss':train_reconstruct_loss,
                    'train_KLD':train_kld_loss,
                    'train_dist':train_distance, 
                    'val_loss': val_loss,
                    'val_reconstruction_Loss':val_reconstruct_loss,
                    'val_KLD':val_kld_loss, 
                    'val_dist':val_distance, 
                    'epoch': epoch
                    })

            self.scheduler.step()
            
            self.early_stopping(val_loss, val_distance, self.model, train_loss)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        return self.model

        
class SupervisedTrainer():
    def __init__(
        self,
        model:nn.Module,
        training_dataset: data.Dataset,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        num_workers: int,
        train_val_split: float = 0.9,
        model_save_path: str = f'model_save_path/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        ):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model

        self.batch_size = batch_size
        train_idx, val_idx = train_test_split(list(range(training_dataset.__len__())), train_size=train_val_split, shuffle=True)
        data_train = data.Subset(training_dataset, train_idx)
        data_val = data.Subset(training_dataset, val_idx)
        self.train_dataloader = data.DataLoader(data_train, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory = True, prefetch_factor=batch_size*2)
        self.val_dataloader = data.DataLoader(data_val, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory = True, prefetch_factor=batch_size*2)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        patience = 5
        self.model_save_path = model_save_path
        self.early_stopping = EarlyStopping(patience=patience, path=model_save_path, model_name = model.__class__.__name__)

        self.num_epochs = num_epochs

        parameter_dict = {
            "dataset": training_dataset.__class__.__name__,
            "model": model.__class__.__name__,
            "model_params": model.get_hyperparamters(),
            "trainer_params": {
                "batch_size": self.batch_size,
                "learning_rate": learning_rate,
                "num_epochs": self.num_epochs,
                "num_workers": num_workers,
                "early_stopping_patience": patience
            }
        }

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        with open(os.path.join(model_save_path, "training_parameters.yaml"), "w") as f:
            yaml.dump(parameter_dict, f)

    def train(self, save_samples: bool = False):
        self.model = self.model.to(self.device)
        
        for epoch in range(self.num_epochs):

            self.model.train()
            train_loss = 0
            train_reconstruct_loss=0
            train_kld_loss=0
            train_distance = 0
            train_classify_loss = 0
            train_classify_acc_items = 0
            train_concept_total_num = 0
            with tqdm(total=len(self.train_dataloader), postfix={'epoch': epoch}) as train_bar:
                for batch_idx, (x, y) in enumerate(self.train_dataloader):
                    x = x.to(self.device)
                    concept_label = y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    classify = [torch.argmax(nn.functional.softmax(o, dim=1),1) for o in output[4]]
                    classify_results = torch.stack(classify, dim=0).transpose(0, 1)
                    train_classify_acc_items += (classify_results == concept_label).sum().item()
                    train_concept_total_num += np.prod(concept_label.shape)
                    loss = self.model.loss_function(*output, concept_label, M_N=0.5, batch_idx=batch_idx)
                    loss['loss'].backward()
                    self.optimizer.step()
                    train_loss += loss['loss'].item()
                    train_reconstruct_loss += loss['Reconstruction_Loss'].item()
                    train_kld_loss += loss['KLD'].item()
                    train_distance += torch.mean(torch.abs(x - output[0])).item()
                    train_classify_loss += loss['Classify_Loss'].item()
                    train_bar.update(1)

            self.scheduler.step()
            train_loss /= self.train_dataloader.dataset.__len__()
            train_reconstruct_loss /= self.train_dataloader.dataset.__len__()
            train_kld_loss /= self.train_dataloader.dataset.__len__()
            train_classify_loss /= self.train_dataloader.dataset.__len__()
            train_distance /= -self.train_dataloader.dataset.__len__()
            train_distance *= 255
            train_bar.set_description(f'avg_train_loss: {train_loss}; avg_train_dist: {train_distance} epoch: {epoch}')

            self.model.eval()
            val_loss = 0
            val_reconstruct_loss=0
            val_kld_loss=0
            val_distance = 0
            val_classify_loss = 0
            val_classify_acc_items = 0
            val_total_num = 0
            with tqdm(total=len(self.val_dataloader)) as val_bar:
                for batch_idx, (x, y) in enumerate(self.val_dataloader):
                    x = x.to(self.device)
                    concept_label = y.to(self.device)
                    output = self.model(x)
                    classify = [torch.argmax(nn.functional.softmax(o, dim=1),1) for o in output[4]]
                    classify_results = torch.stack(classify, dim=0).transpose(0, 1)
                    val_classify_acc_items += (classify_results == concept_label).sum().item()
                    val_total_num += np.prod(concept_label.shape)
                    loss = self.model.loss_function(*output, concept_label, batch_idx=batch_idx)
                    val_loss += loss['loss'].item()
                    val_reconstruct_loss += loss['Reconstruction_Loss'].item()
                    val_kld_loss += loss['KLD'].item()
                    val_distance += torch.mean(torch.abs(x - output[0])).item()
                    val_classify_loss += loss['Classify_Loss'].item()
                    val_bar.update(1)

                    if save_samples and batch_idx == 0:
                        save_path = os.path.join(self.early_stopping.path, "images")
                        os.makedirs(save_path, exist_ok=True)
                        plot_generation_with_input(
                            x[:3],
                            output[0][:3],
                            save_path=os.path.join(save_path, f"epoch_{epoch}.png")
                        )

            val_loss /= self.val_dataloader.dataset.__len__()
            val_reconstruct_loss /= self.val_dataloader.dataset.__len__()
            val_kld_loss /= self.val_dataloader.dataset.__len__()
            val_classify_loss /= self.val_dataloader.dataset.__len__()
            val_distance /= -self.val_dataloader.dataset.__len__()
            val_distance *= 255
            val_bar.set_description(f'avg_val_loss: {val_loss}; avg_val_dist: {val_distance}; epoch: {epoch}.')
            wandb.log({
                    'train_loss': train_loss, 
                    'train_reconstruction_Loss':train_reconstruct_loss,
                    'train_KLD':train_kld_loss,
                    'train_dist':train_distance, 
                    'train_classify_loss': train_classify_loss,
                    'train_acc': train_classify_acc_items/train_concept_total_num,
                    'val_loss': val_loss,
                    'val_reconstruction_Loss':val_reconstruct_loss,
                    'val_KLD':val_kld_loss, 
                    'val_dist':val_distance,
                    'val_classify_loss': val_classify_loss,
                    'val_acc': val_classify_acc_items/val_total_num,
                    'epoch': epoch
                    })

            self.scheduler.step()
            
            self.early_stopping(val_loss, val_distance, self.model, train_loss)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        return self.model


class SupervisedEvaluator():
    def __init__(
        self,
        model:nn.Module,
        testing_dataset: data.Dataset,
        batch_size: int,
        num_workers: int,
        ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.test_dataloader = data.DataLoader(testing_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory = True, prefetch_factor=batch_size*2)

    def evaluate(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        test_loss=0
        reconstruct_loss=0
        kld_loss=0
        test_distance=0
        classify_loss = 0
        classify_acc_items = 0
        total_num = 0
        with tqdm(total=len(self.test_dataloader)) as test_bar:
            for batch_idx, (x, y) in enumerate(self.test_dataloader):
                    x = x.to(self.device)
                    concept_label = y.to(self.device)
                    output = self.model(x)
                    loss = self.model.loss_function(*output, concept_label, batch_idx=batch_idx)
                    classify = [torch.argmax(nn.functional.softmax(o, dim=1),1) for o in output[4]]
                    classify_results = torch.stack(classify, dim=0).transpose(0, 1)
                    classify_acc_items += (classify_results == concept_label).sum().item()
                    total_num += np.prod(concept_label.shape)
                    test_loss += loss['loss'].item()
                    reconstruct_loss += loss['Reconstruction_Loss'].item()
                    kld_loss += loss['KLD'].item()
                    test_distance += torch.mean(torch.abs(x - output[0])).item()
                    classify_loss += loss['Classify_Loss'].item()
                    test_bar.update(1)

            test_loss /= self.test_dataloader.dataset.__len__()
            reconstruct_loss /= self.test_dataloader.dataset.__len__()
            kld_loss /= self.test_dataloader.dataset.__len__()
            test_distance /= -self.test_dataloader.dataset.__len__() 
            test_distance *= 255
            test_bar.set_description(f'test_loss: {test_loss}; test_dist: {test_distance}.')
            wandb.log({
                    'test_loss': test_loss, 
                    'Reconstruction_Loss':reconstruct_loss, 
                    'KLD':kld_loss,
                    'test_distance': test_distance
                    })


class Evaluator():
    def __init__(
        self,
        model:nn.Module,
        testing_dataset: data.Dataset,
        batch_size: int,
        num_workers: int,
        ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.test_dataloader = data.DataLoader(testing_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory = True, prefetch_factor=batch_size*2)

    def evaluate(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        test_loss=0
        reconstruct_loss=0
        kld_loss=0
        test_distance=0
        with tqdm(total=len(self.test_dataloader)) as test_bar:
            for batch_idx, (x, y) in enumerate(self.test_dataloader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)
                    loss = self.model.loss_function(*output, M_N=0.5, batch_idx=batch_idx)
                    test_loss += loss['loss'].item()
                    reconstruct_loss += loss['Reconstruction_Loss'].item()
                    kld_loss += loss['KLD'].item()
                    test_distance += torch.mean(torch.abs(x - output[0])).item()
                    test_bar.update(1)

            test_loss /= self.test_dataloader.dataset.__len__()
            reconstruct_loss /= self.test_dataloader.dataset.__len__()
            kld_loss /= self.test_dataloader.dataset.__len__()
            test_distance /= -self.test_dataloader.dataset.__len__() 
            test_distance *= 255
            test_bar.set_description(f'test_loss: {test_loss}; test_dist: {test_distance}.')
            wandb.log({
                    'test_loss': test_loss, 
                    'Reconstruction_Loss':reconstruct_loss, 
                    'KLD':kld_loss,
                    'test_distance': test_distance
                    })


class AugementedLagrangianTrainer():
    def __init__(
        self,
        model:CausalConceptVAE,
        training_dataset: data.Dataset,
        test_dataset: data.Dataset,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        num_workers: int,
        dataset_name: str,
        model_save_path: str = f'model_save_path/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
        lagrangian_mu_init: float = 1e-3, 
        lagrangian_lambda_init: float = 0.
        ):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model

        self.batch_size = batch_size
        self.train_dataloader = data.DataLoader(training_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory = True, prefetch_factor=batch_size*2)
        self.val_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory = True, prefetch_factor=batch_size*2)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        patience = 10
        self.model_save_path = model_save_path
        self.early_stopping = EarlyStopping(patience=patience, path=model_save_path, model_name = model.__class__.__name__)

        self.num_epochs = num_epochs

        self.lagrangian_mu_init = lagrangian_mu_init
        self.lagrangian_lambda_init = lagrangian_lambda_init
        
        wandb.init(
            project="coca_workflow",
            name = f"{model.__class__.__name__}_{dataset_name}"
        )

        parameter_dict = {
            "dataset": training_dataset.__class__.__name__,
            "model": model.__class__.__name__,
            "model_params": model.get_hyperparamters(),
            "trainer_params": {
                "batch_size": self.batch_size,
                "learning_rate": learning_rate,
                "num_epochs": self.num_epochs,
                "num_workers": num_workers,
                "early_stopping_patience": patience
            },
            "augmented_lagrangian_params": {
                "lambda_init": self.lagrangian_lambda_init,
                "mu_init": self.lagrangian_mu_init
            }
        }

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        with open(os.path.join(model_save_path, "training_parameters.yaml"), "w") as f:
            yaml.dump(parameter_dict, f)

    def augemented_lagrangian_training(self):
        # construction_loss + 0.5 * mu * h_loss ** 2 + lamb * h_loss + KL_loss
        # Augmented Lagrangian stuff
        mu = self.lagrangian_mu_init
        lamb = self.lagrangian_lambda_init

        val_losses = []
        hs = []

        self.model = self.model.to(self.device)

        for epoch in range(self.num_epochs):

            # <---------------Model Training---------------->
            self.model.train()

            train_loss = 0
            train_reconstruct_loss=0
            train_kld_loss=0
            train_distance = 0
            train_classify_loss = 0
            train_classify_acc_items = 0
            train_concept_total_num = 0
            train_acyclic_loss = 0
            training_single_h = 0

            with tqdm(total=len(self.train_dataloader), postfix={'epoch': epoch}) as train_bar:
                for batch_idx, items in enumerate(self.train_dataloader):
                    x = items[0].to(self.device)
                    concept_label = items[2].to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    classify = [torch.argmax(o,1) for o in output[4]]
                    classify_results = torch.stack(classify, dim=-1)
                    train_classify_acc_items += (classify_results == concept_label).sum().item()
                    train_concept_total_num += np.prod(concept_label.shape)
                    loss = self.model.loss_function(*output, concept_label)

                    h_loss = self.model.h_loss(self.model.get_causal_adjacency_matrix())

                    loss['acyclic'] = 0.5 * mu * h_loss ** 2 + lamb * h_loss
                    loss['loss'] = loss['loss'] + loss['acyclic']

                    loss['loss'].backward()
                    self.optimizer.step()
                    train_loss += loss['loss'].item()
                    train_reconstruct_loss += loss['Reconstruction_Loss'].item()
                    train_kld_loss += loss['KLD'].item()
                    train_acyclic_loss += loss['acyclic'].item()
                    train_distance += torch.mean(torch.abs(x - output[0])).item()
                    train_classify_loss += loss['Classify_Loss'].item()
                    training_single_h += h_loss.item()
                    train_bar.update(1)

            train_loss /= self.train_dataloader.dataset.__len__()
            train_reconstruct_loss /= self.train_dataloader.dataset.__len__()
            train_kld_loss /= self.train_dataloader.dataset.__len__()
            train_classify_loss /= self.train_dataloader.dataset.__len__()
            train_acyclic_loss /= self.train_dataloader.dataset.__len__()
            train_distance /= -self.train_dataloader.dataset.__len__()
            train_distance *= 255
            training_single_h /= self.train_dataloader.dataset.__len__()
            train_bar.set_description(f'avg_train_loss: {train_loss}; avg_train_dist: {train_distance} epoch: {epoch}')

            # <---------------Model Validation---------------->
            self.model.eval()
            val_loss = 0
            val_reconstruct_loss=0
            val_kld_loss=0
            val_distance = 0
            val_classify_loss = 0
            val_classify_acc_items = 0
            val_total_num = 0
            val_acyclic_loss = 0
            val_single_h = 0
            with tqdm(total=len(self.val_dataloader)) as val_bar:
                for batch_idx, items in enumerate(self.val_dataloader):
                    x = items[0].to(self.device)
                    concept_label = items[2].to(self.device)
                    output = self.model(x)
                    classify = [torch.argmax(o,1) for o in output[4]]
                    classify_results = torch.stack(classify, dim=-1)
                    val_classify_acc_items += (classify_results == concept_label).sum().item()
                    val_total_num += np.prod(concept_label.shape)
                    loss = self.model.loss_function(*output, concept_label)
                    h_loss = self.model.h_loss(self.model.get_causal_adjacency_matrix())
                    loss['acyclic'] = 0.5 * mu * h_loss ** 2 + lamb * h_loss
                    loss['loss'] = loss['loss'] + loss['acyclic']

                    val_loss += loss['loss'].item()
                    val_reconstruct_loss += loss['Reconstruction_Loss'].item()
                    val_kld_loss += loss['KLD'].item()
                    val_acyclic_loss += loss['acyclic'].item()
                    val_distance += torch.mean(torch.abs(x - output[0])).item()
                    val_classify_loss += loss['Classify_Loss'].item()
                    val_single_h += h_loss.item()
                    val_bar.update(1)

            val_loss /= self.val_dataloader.dataset.__len__()
            val_reconstruct_loss /= self.val_dataloader.dataset.__len__()
            val_kld_loss /= self.val_dataloader.dataset.__len__()
            val_classify_loss /= self.val_dataloader.dataset.__len__()
            val_acyclic_loss /= self.val_dataloader.dataset.__len__()
            val_distance /= -self.val_dataloader.dataset.__len__()
            val_distance *= 255
            val_single_h /= self.val_dataloader.dataset.__len__()
            val_losses.append(val_single_h)
            # val_losses.append(val_acyclic_loss)
            val_bar.set_description(f'avg_val_loss: {val_loss}; avg_val_dist: {val_distance}; epoch: {epoch}.')

        
            # <---------------updata augmented lagrangian parameters---------------->

            if len(val_losses) >= 3:
                t0, t_half, t1 = val_losses[-3], val_losses[-2], val_losses[-1]
                if not (min(t0, t1) < t_half < max(t0, t1)):
                    delta_lambda = -np.inf
                else:
                    delta_lambda = (t1 - t0)
            else:
                delta_lambda = -np.inf

            if training_single_h > 1e-8:
                if abs(delta_lambda) < 1e-8 or delta_lambda > 0:
                    lamb += mu * training_single_h

                    hs.append(training_single_h)

                    if (len(hs) >= 2) and (hs[-1] > hs[-2] * 0.9):
                        mu *= 10

            # <---------------Logging---------------->
            wandb.log({
                    'train_loss': train_loss, 
                    'train_reconstruction_Loss':train_reconstruct_loss,
                    'train_KLD':train_kld_loss,
                    'train_dist':train_distance, 
                    'train_classify_loss': train_classify_loss,
                    'train_acc': train_classify_acc_items/train_concept_total_num,
                    'train_acyclic_loss': train_acyclic_loss,
                    'train_single_h': training_single_h,
                    'val_loss': val_loss,
                    'val_reconstruction_Loss':val_reconstruct_loss,
                    'val_KLD':val_kld_loss, 
                    'val_dist':val_distance,
                    'val_classify_loss': val_classify_loss,
                    'val_acc': val_classify_acc_items/val_total_num,
                    'val_acyclic_loss': val_acyclic_loss,
                    'val_single_h': val_single_h,
                    'mu': mu,
                    'lamb': lamb,
                    'epoch': epoch,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'num_masked_items': (self.model.causal_adjacency_mask == 0).sum().item()
                    })

            self.scheduler.step()
            
            self.early_stopping(val_loss, val_distance, self.model, train_loss)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        return self.model