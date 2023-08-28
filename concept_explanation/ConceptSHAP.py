import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pickle
import numpy as np
import math
from itertools import chain, combinations
from tqdm import tqdm
from data.pvr_concepts_dataset_generation import FeatureMapsDataset
from model.pytorchtools import EarlyStopping
import os


class ConceptNet(nn.Module):

    def __init__(self, n_concepts, train_embeddings, original_model:nn.Module, bottleneck:str, example_input_for_original_model:torch.tensor):
        super(ConceptNet, self).__init__()
        embedding_dim = train_embeddings.shape[1]
        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(embedding_dim, n_concepts), requires_grad=True)
        self.n_concepts = n_concepts
        self.train_embeddings = train_embeddings.transpose(0, 1) # (dim, all_data_size)

        self.original_model = original_model
        self.bottleneck = bottleneck
        for name,module in self.original_model.named_modules():
        #     if is_found_layer:
        #         self.bottleneck = name
        #         break
             if name == self.bottleneck:
                module.register_forward_hook(self.forward_hook)
        #         is_found_layer = True

        # if self.bottleneck == None:
        #     raise Exception(f"Can not found {bottleneck} next layer")
        self.original_model_input_sample = example_input_for_original_model
        self.modified_feature_map = None


    def forward_hook(self, module, input, output):
        return self.modified_feature_map

    def init_concept(self, embedding_dim, n_concepts):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept

    def forward(self, train_embedding, topk):
        """
        train_embedding: shape (bs, embedding_dim)
        """
        # calculating projection of train_embedding onto the concept vector space
        proj_matrix = (self.concept @ torch.inverse((self.concept.T @ self.concept))) @ self.concept.T # (embedding_dim x embedding_dim)
        proj = proj_matrix @ train_embedding.T  # (embedding_dim x batch_size)
        print(proj_matrix.shape, self.concept.shape, train_embedding.shape, proj.shape)

        # passing projected activations through rest of model
        self.modified_feature_map = proj.T
        y_pred = self.original_model(self.original_model_input_sample)
        self.modified_feature_map = train_embedding
        orig_pred = self.original_model(self.original_model_input_sample)
        print(y_pred,orig_pred)

        # Calculate the regularization terms as in new version of paper
        k = topk # this is a tunable parameter

        ### calculate first regularization term, to be maximized
        # 1. find the top k nearest neighbour
        all_concept_knns = []
        for concept_idx in range(self.n_concepts):
            c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)

            # euc dist
            distance = torch.norm(self.train_embeddings - c, dim=0) # (num_total_activations)
            knn = distance.topk(k, largest=False)
            indices = knn.indices # (k)
            knn_activations = self.train_embeddings[:, indices] # (activation_dim, k)
            all_concept_knns.append(knn_activations)

        # 2. calculate the avg dot product for each concept with each of its knn
        L_sparse_1_new = 0.0
        for concept_idx in range(self.n_concepts):
            c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)
            c_knn = all_concept_knns[concept_idx] # knn for c
            dot_prod = torch.sum(c * c_knn) / k # avg dot product on knn
            L_sparse_1_new += dot_prod
        L_sparse_1_new = L_sparse_1_new / self.n_concepts

        ### calculate Second regularization term, to be minimized
        all_concept_dot = self.concept.T @ self.concept
        mask = torch.eye(self.n_concepts).cuda() * -1 + 1 # mask the i==j positions
        L_sparse_2_new = torch.mean(all_concept_dot * mask)

        norm_metrics = torch.mean(all_concept_dot * torch.eye(self.n_concepts).cuda())

        concept_pred = self.concept @ train_embedding

        return orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new, [norm_metrics], concept_pred

    def loss(self, train_embedding, train_y_true, regularize, doConceptSHAP, l_1, l_2, topk):
        """
        This function will be called externally to feed data and get the loss
        """
        # Note: it is important to MAKE SURE L2 GOES DOWN! that will let concepts separate from each other

        orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new, metrics, concept_pred = self.forward(train_embedding, topk)

        ce_loss = nn.CrossEntropyLoss()
        loss_new = ce_loss(y_pred, train_y_true)
        pred_loss = torch.mean(loss_new)

        # completeness score
        def n(y_pred):
            orig_correct = torch.sum(train_y_true == torch.argmax(orig_pred, axis=1))
            new_correct = torch.sum(train_y_true == torch.argmax(y_pred, axis=1))
            return torch.div(new_correct - (1/self.n_concepts), orig_correct - (1/self.n_concepts))

        completeness = n(y_pred)

        conceptSHAP = []
        if doConceptSHAP:
            def proj(concept):
                proj_matrix = (concept @ torch.inverse((concept.T @ concept))) \
                              @ concept.T  # (embedding_dim x embedding_dim)
                proj = proj_matrix @ train_embedding.T  # (embedding_dim x batch_size)

                # passing projected activations through rest of model
                self.modified_feature_map = proj.T
                return self.original_model(self.original_model_input_sample)

            # shapley score (note for n_concepts > 10, this is very inefficient to calculate)
            c_id = np.asarray(list(range(len(self.concept.T))))
            for idx in c_id:
                exclude = np.delete(c_id, idx)
                subsets = np.asarray(self.powerset(list(exclude)))
                sum = 0
                for subset in subsets:
                    # score 1:
                    c1 = subset + [idx]
                    concept = np.take(self.concept.T.detach().cpu().numpy(), np.asarray(c1), axis=0)
                    concept = torch.from_numpy(concept).T
                    pred = proj(concept.cuda())
                    score1 = n(pred)

                    # score 2:
                    c1 = subset
                    if c1 != []:
                        concept = np.take(self.concept.T.detach().cpu().numpy(), np.asarray(c1), axis=0)
                        concept = torch.from_numpy(concept).T
                        pred = proj(concept.cuda())
                        score2 = n(pred)
                    else: score2 = torch.tensor(0)

                    norm = (math.factorial(len(c_id) - len(subset) - 1) * math.factorial(len(subset))) / \
                           math.factorial(len(c_id))
                    sum += norm * (score1.data.item() - score2.data.item())
                conceptSHAP.append(sum)

        if regularize:
            final_loss = pred_loss + (l_1 * L_sparse_1_new * -1) + (l_2 * L_sparse_2_new)
        else:
            final_loss = pred_loss

        return completeness, conceptSHAP, final_loss, pred_loss, L_sparse_1_new, L_sparse_2_new, metrics
        # return final_loss

    def powerset(self, iterable):
        "powerset([1,2,3]) --> [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]"
        s = list(iterable)
        pset = chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))
        return [list(i) for i in list(pset)]


class ConceptSHAP(object):

    def __init__(self, n_concepts:int, input_exmaple:torch.tensor, original_model:nn.modules, bottleneck:str) -> None:
        # original_model is the explained deep learning model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concept_model = ConceptNet(n_concepts, input_exmaple).to(self.device)
        self.optimizer = torch.optim.Adam(self.concept_model.parameters(),lr=1e-3)
        self.num_epoch = 10
        self.batch_size = 2
        self.original_model = original_model
        self.bottleneck = bottleneck


    def train(self, x, y):
        dataset = FeatureMapsDataset(x,y)
        train_dataset,val_dataset = random_split(dataset, [int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=6, pin_memory = True, prefetch_factor=self.batch_size*2)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=6, pin_memory = True, prefetch_factor=self.batch_size*2)

        for epoch_index in tqdm(range(self.num_epoch)):
            for train_embeddings_narrow, train_y_true_narrow in train_dataloader:
                final_loss = self.model.loss(train_embeddings_narrow,
                                                                                       train_y_true_narrow,
                                                                                       doConceptSHAP=True,
                                                                                       l_1=l_1, l_2=l_2, topk=topk)

                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()

        for val_embeddings_narrow, val_y_true_narrow in val_dataloader:
            final_loss = self.model.loss(train_embeddings_narrow,
                                                                                       train_y_true_narrow,
                                                                                       doConceptSHAP=True,
                                                                                       l_1=l_1, l_2=l_2, topk=topk)

    def save_model(self, concepts, save_path):
        save_dict = {
            'concepts': concepts,
            'bottleneck': self.bottleneck,
            'accuracies': self.accuracies,
            'saved_path': save_path
        }
        if self.save_path is not None:
            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)
        else:
            print('save_path is None. Not saving anything')

class SimplifiedConceptModel(nn.Module):
    def __init__(self, n_concepts, train_embeddings, original_model:nn.Module, bottleneck:str, example_input_for_original_model:torch.tensor, device:torch.device):
        super(SimplifiedConceptModel, self).__init__()
        embedding_dim = train_embeddings.view(train_embeddings.shape[0],-1).shape[1]
        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(embedding_dim, n_concepts), requires_grad=True)
        self.n_concepts = n_concepts
        self.train_embeddings = train_embeddings.transpose(0, 1) # (dim, all_data_size)

        self.original_model = original_model.eval().to(device)
        self.bottleneck = bottleneck
        self.add_hook()
        self.original_model_input_sample = torch.unsqueeze(example_input_for_original_model, dim=0).to(device)
        self.modified_feature_map = None
        self.device = device

    def init_concept(self, embedding_dim, n_concepts):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept

    def forward_hook(self, module, input, output):
        return self.modified_feature_map

    def forward(self, train_embedding):
        origninal_embedding = train_embedding
        train_embedding = torch.reshape(train_embedding,(train_embedding.shape[0],-1)).to(self.device)
        concept_pred = train_embedding @ self.concept

        # calculating projection of train_embedding onto the concept vector space

        a = self.concept @ torch.inverse((self.concept.T @ self.concept))
        proj = a @ (self.concept.T @ train_embedding.T) # (embedding_dim x embedding_dim) (embedding_dim x batch_size)

        # passing projected activations through rest of model

        
        self.modified_feature_map = torch.reshape(proj.T, origninal_embedding.shape) 
        y_pred = self.original_model(self.original_model_input_sample)
        self.modified_feature_map = origninal_embedding

        orig_pred = self.original_model(self.original_model_input_sample)
        orig_pred = torch.argmax(orig_pred, dim=1)

        # print(concept_pred.shape, orig_pred.shape)
        return concept_pred, y_pred, orig_pred

    def add_hook(self):
        for name,module in self.original_model.named_modules():
             if name == self.bottleneck:
                self.hook = module.register_forward_hook(self.forward_hook)
                break

    def remove_hook(self):
        self.hook.remove()



class SimplifiedConceptSHAP(object):
    def __init__(self, n_concepts, train_embeddings, original_model:nn.Module, bottleneck:str, example_input_for_original_model:torch.tensor, save_path, concept):
        super(SimplifiedConceptSHAP, self).__init__()
        # random init using uniform dist
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concept_model = SimplifiedConceptModel(n_concepts, train_embeddings, original_model.eval(), bottleneck, example_input_for_original_model, self.device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.save_path = save_path
        self.batch_size = 5
        self.num_epoch = 100
        self.optimizer = torch.optim.Adam(self.concept_model.parameters(),lr=1e-3)
        self.concept = concept
        

    def loss(self, train_embedding, concept_true):
        concept_pred, y_pred, orig_pred = self.concept_model(train_embedding)
        loss_new = self.ce_loss(concept_pred, concept_true) + self.ce_loss(y_pred, orig_pred)

        return loss_new, concept_pred


    def train(self, x, y):
        dataset = FeatureMapsDataset(x,y)
        train_dataset,val_dataset = random_split(dataset, [int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=6, pin_memory = True, prefetch_factor=self.batch_size*2)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=6, pin_memory = True, prefetch_factor=self.batch_size*2)
        earlystop = EarlyStopping(patience=5, path=".".join(self.save_path.split(".")[:-1])+".pt")

        self.concept_model = self.concept_model.train().to(self.device)
        for epoch_index in range(self.num_epoch):
            with tqdm(total=len(train_dataloader)) as tbar:
                for train_embeddings_narrow, train_concept_true_narrow in train_dataloader:
                    train_embeddings_narrow = train_embeddings_narrow.to(self.device)
                    train_concept_true_narrow = train_concept_true_narrow.to(self.device)
                    final_loss,_ = self.loss(train_embeddings_narrow, train_concept_true_narrow)

                    self.optimizer.zero_grad()
                    final_loss.backward()
                    self.optimizer.step()
                    tbar.update(1)

            correct_num = 0
            number_samples = len(val_dataloader.dataset)
            all_loss = 0 
            with tqdm(total=len(val_dataloader)) as tbar:
                for val_embeddings_narrow, val_y_true_narrow in val_dataloader:
                    val_embeddings_narrow = val_embeddings_narrow.to(self.device)
                    val_y_true_narrow = val_y_true_narrow.to(self.device)
                    l, concept_pred=self.loss(val_embeddings_narrow, val_y_true_narrow)

                    all_loss += l
                    
                    correct_num += torch.eq(torch.argmax(concept_pred,dim=1),val_y_true_narrow).sum().float().cpu().item()
                    tbar.update(1)
            
            self.acc = correct_num/number_samples

            earlystop(all_loss/number_samples, self.acc, self.concept_model)

            if earlystop.early_stop:
                print("Early stopping")
                break

        print(f"Acc {earlystop.acc}")
        self.earlystop = earlystop
        self.save_model()
        self.concept_model.remove_hook()

    def save_model(self):
        save_dict = {
            'concepts': self.concept,
            'accuracies': self.earlystop.acc,
            'saved_path': self.save_path
        }
        if self.save_path is not None:
            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)
        else:
            print('save_path is None. Not saving anything')
        
        

