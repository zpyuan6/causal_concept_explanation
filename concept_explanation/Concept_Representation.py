from sklearn.decomposition import IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.utils.data as data
import os
import pickle

from data.pvr_concepts_dataset_generation import FeatureMapsDataset

class ConceptBasedCausalVariable(object):
    """TREECAV class contains methods for concept activation vector (TREECAV).

    TREECAV represents semenatically meaningful vector directions in
    network's embeddings (bottlenecks).
    """
    def __init__(self, concepts, bottleneck, hparams, concept_class_num, x, save_path=None):
        """Initialize TREECAV class.

        Args:
          concepts: set of concepts used for TREECAV
          bottleneck: the bottleneck used for TREECAV
          hparams: a parameter used to learn TREECAV
            {   
                dimensionality_reduction:''
            }
          save_path: where to save this TREECAV
        """
        self.concepts = concepts
        self.concept_class_num = concept_class_num
        self.dr_dimention_num = min(int(x.view(x.shape[0],-1).shape[-1]/2),100)
        self.bottleneck = bottleneck
        self.hparams = hparams
        self.save_path = save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.hparams["dimensionality_reduction"] == "LinearDiscriminantAnalysis":
            self.concepts_model = nn.Linear(self.concept_class_num-1, self.concept_class_num).train().to(self.device)
        else:
            self.concepts_model = nn.Linear(self.dr_dimention_num, self.concept_class_num).train().to(self.device)
        torch.nn.init.xavier_uniform_(self.concepts_model.weight)
        torch.nn.init.constant_(self.concepts_model.bias,0)
        self.batch_size = 5
        self.num_epoch = 5
        self.loss = nn.CrossEntropyLoss()

        if self.hparams["dimensionality_reduction"] == "kernelpca":
            print("kernal pca,", self.dr_dimention_num)
            self.dr_methods = KernelPCA(n_components=self.dr_dimention_num, kernel="poly")
        elif self.hparams["dimensionality_reduction"] == "incrementalpca":
            self.dr_methods = IncrementalPCA(n_components=self.dr_dimention_num, batch_size=self.dr_dimention_num)
        elif self.hparams["dimensionality_reduction"] == "sparsepca":
            self.dr_methods = SparsePCA(n_components=self.dr_dimention_num)
        elif self.hparams["dimensionality_reduction"] == "truncatedSVD":
            self.dr_methods = TruncatedSVD(n_components=self.dr_dimention_num)
        elif self.hparams["dimensionality_reduction"] == "LinearDiscriminantAnalysis":
            self.dr_methods = LinearDiscriminantAnalysis(n_components=self.concept_class_num-1)
        elif self.hparams["dimensionality_reduction"] == "LinearLayer":
            self.dr_methods = nn.Linear(x.view(x.shape[0],-1).shape[-1], self.dr_dimention_num).train().to(self.device)
        elif self.hparams["dimensionality_reduction"] == "t-SNE":
            self.dr_methods = TSNE(n_components=self.dr_dimention_num)
        else:
            raise ValueError('Invalid hparams.dimensionality_reduction: {}'.format(self.hparams["dimensionality_reduction"]))


        if self.hparams["dimensionality_reduction"]!="LinearLayer":
            self.optimiser = torch.optim.Adam(self.concepts_model.parameters(),lr=1e-3)
        else:
            self.optimiser = torch.optim.Adam([
                {'params':self.concepts_model.parameters(),'lr':1e-3},
                {'params':self.dr_methods.parameters(),'lr':1e-3}] )

    def train(self, x, label):
        """Train the TREECAVs from the activations.

        Args:
          x: is a list of numpy as input of concept, feature maps of explained model
                {'concept1':{'bottleneck name1':[...act array...],
                             'bottleneck name2':[...act array...],...
                 'concept2':{'bottleneck name1':[...act array...],
        Raises:
          ValueError: if the model_type in hparam is not compatible.
        """

        print('training ConceptBasedCausalVariable')
        dataset = FeatureMapsDataset(x,label)
        self.accuracies = self._train_lm(dataset, x, label)
        self._save_model()


    def _train_lm(self, dataset:data.Dataset, x:torch.tensor, y:torch.tensor):
        """Train a model to get TREECAVs.

        Modifies lm by calling the lm.fit functions. The TREECAVs coefficients are then
        in lm._coefs.

        Args:
          lm: An sklearn linear_model object. Can be linear regression or
            logistic regression. Must support .fit and ._coef.
          x: An array of training data of shape [num_data, data_dim]
          y: An array of integer labels of shape [num_data]
          labels2text: Dictionary of text for each label.

        Returns:
          Dictionary of accuracies of the TREECAVs.

        """
        # x = []
        # y = []
        # for input, label in dataset:
        #     x.append(input.numpy())
        #     y.append(label.numpy())
        # x = np.array(x)
        # y = np.array(y)

        if self.hparams["dimensionality_reduction"] == "LinearDiscriminantAnalysis":
            self.dr_methods.fit(x.view(x.shape[0],-1).numpy(), y.numpy())
        elif self.hparams["dimensionality_reduction"] == "LinearLayer":
            self.dr_methods.train() 
        else:
            self.dr_methods.fit(x.view(x.shape[0],-1).numpy())


        train_dataset,val_dataset = data.random_split(dataset, [int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
        train_dataloader = data.DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=6, pin_memory = True, prefetch_factor=self.batch_size*2)
        val_dataloader = data.DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=6, pin_memory = True, prefetch_factor=self.batch_size*2)

        self.concepts_model.train()
        for epoch_index in range(self.num_epoch):
            with tqdm(total=len(train_dataloader)) as tbar:
                for train_features, concept_labels in train_dataloader:
                    
                    train_features = train_features.view(train_features.shape[0],-1)
                    if self.hparams["dimensionality_reduction"] == "LinearLayer":
                        train_features = train_features.to(self.device)
                        dr_features = self.dr_methods(train_features)
                    else:
                        train_features = train_features.numpy()
                        dr_features = self.dr_methods.transform(train_features)
                        dr_features = torch.from_numpy(dr_features).to(self.device)
                    concept_labels = concept_labels.to(self.device)
                    concept_predict = self.concepts_model(dr_features.float())
                    l = self.loss(concept_predict, concept_labels)

                    self.optimiser.zero_grad()
                    l.backward()
                    self.optimiser.step()
                    tbar.update(1)

        correct_num = 0
        number_samples = len(val_dataloader.dataset)
        if self.hparams["dimensionality_reduction"] == "LinearLayer":
            self.dr_methods.eval()
        self.concepts_model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_dataloader)) as tbar:
                for train_features, concept_labels in val_dataloader:
                    with torch.no_grad():
                        train_features = train_features.view(train_features.shape[0],-1)
                        if self.hparams["dimensionality_reduction"] == "LinearLayer":
                            dr_features = self.dr_methods(train_features.to(self.device))
                        else:
                            train_features = train_features.numpy()
                            dr_features = self.dr_methods.transform(train_features)
                            dr_features = torch.from_numpy(dr_features).to(self.device)
                        concept_labels = concept_labels.to(self.device)
                        concept_predict = self.concepts_model(dr_features.float())

                        correct_num += torch.eq(torch.argmax(concept_predict,dim=1),concept_labels).sum().float().cpu().item()
                        tbar.update(1)

        self.acc = correct_num/number_samples
        print(f"Acc {self.acc}")

    def _save_model(self):
        """Save a dictionary of this TREECAVs to a pickle."""
        save_dict = {
            'concepts': self.concepts,
            'bottleneck': self.bottleneck,
            'dr_methods': self.dr_methods,
            'concepts_model': self.concepts_model,
            'hparams': self.hparams,
            'accuracies': self.acc,
            'saved_path': self.save_path
        }
        if self.save_path is not None:
            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)
        else:
            print('save_path is None. Not saving anything')

    # def load_from_txt(self, save_path=None):
    #     if not save_path==None:
    #         self.save_path = save_path
        
    #     if not os.path.exists(self.save_path):
    #         raise Exception(f"Can not found file in path {self.save_path}")

    #     with open(self.save_path, 'rb') as file:
    #         cav = pickle.load(file)
    #         self.concepts = cav.concepts
    #         self.bottleneck = cav.bottleneck
    #         self.hparams = cav.hparams
    #         self.accuracies = cav.accuracies
    #         self.lm = cav.lm