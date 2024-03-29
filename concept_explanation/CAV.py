import torch
import pickle
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os


class CAV(object):
    """CAV class contains methods for concept activation vector (CAV).

    CAV represents semenatically meaningful vector directions in
    network's embeddings (bottlenecks).
    """
    def __init__(self, concepts, bottleneck, hparams, save_path=None):
        """Initialize CAV class.

        Args:
          concepts: set of concepts used for CAV
          bottleneck: the bottleneck used for CAV
          hparams: a parameter used to learn CAV
            {
                model_type:''
                alpha:''
            }
          save_path: where to save this CAV
        """
        self.concepts = concepts
        self.bottleneck = bottleneck
        self.hparams = hparams
        self.save_path = save_path

    def train(self, x, label):
        """Train the CAVs from the activations.

        Args:
          acts: is a dictionary of activations. In particular, acts takes for of
                {'concept1':{'bottleneck name1':[...act array...],
                             'bottleneck name2':[...act array...],...
                 'concept2':{'bottleneck name1':[...act array...],
        Raises:
          ValueError: if the model_type in hparam is not compatible.
        """

        print('training with alpha={}'.format(self.hparams["alpha"]))

        if self.hparams["model_type"] == 'linear':
            lm = linear_model.SGDClassifier(alpha=self.hparams["alpha"], tol=1e-3, max_iter=1000)
        elif self.hparams["model_type"] == 'logistic':
            lm = linear_model.LogisticRegression()
        else:
            raise ValueError('Invalid hparams.model_type: {}'.format(self.hparams["model_type"]))

        self.accuracies = self._train_lm(lm, x, label)
        if len(lm.coef_) == 1:
            # if there were only two labels, the concept is assigned to label 0 by
            # default. So we flip the coef_ to reflect this.
            self.cavs = [-1 * lm.coef_[0], lm.coef_[0]]
        else:
            self.cavs = [c for c in lm.coef_]
        self._save_cavs()

    def _train_lm(self, lm, x:torch.tensor, y:torch.tensor):
        """Train a model to get CAVs.

        Modifies lm by calling the lm.fit functions. The cav coefficients are then
        in lm._coefs.

        Args:
          lm: An sklearn linear_model object. Can be linear regression or
            logistic regression. Must support .fit and ._coef.
          x: An array of training data of shape [num_data, data_dim]
          y: An array of integer labels of shape [num_data]
          labels2text: Dictionary of text for each label.

        Returns:
          Dictionary of accuracies of the CAVs.

        """
        # x = []
        # y = []
        # for input, label in dataset:
        #     x.append(input.numpy())
        #     y.append(label.numpy())
        # x = np.array(x)
        # y = np.array(y)
        x = x.view(x.shape[0],-1).numpy()
        y = y.numpy()

        print(x.shape, y.shape, type(x), type(y))

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, stratify=y)
        # if you get setting an array element with a sequence, chances are that your
        # each of your activation had different shape - make sure they are all from
        # the same layer, and input image size was the same
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_test)
        # get acc for each class.
        num_classes = max(y) + 1
        acc = {}
        num_correct = 0
        for class_id in range(num_classes):
            # get indices of all test data that has this class.
            idx = (y_test == class_id)
            acc[class_id] = metrics.accuracy_score(
                y_pred[idx], y_test[idx])
            # overall correctness is weighted by the number of examples in this class.
            num_correct += (sum(idx) * acc[class_id])
        acc['overall'] = float(num_correct) / float(len(y_test))
        print('acc per class %s' % (str(acc)))
        return acc

    def _save_cavs(self):
        """Save a dictionary of this CAV to a pickle."""
        save_dict = {
            'concepts': self.concepts,
            'bottleneck': self.bottleneck,
            'hparams': self.hparams,
            'accuracies': self.accuracies,
            'cavs': self.cavs,
            'saved_path': self.save_path
        }
        if self.save_path is not None:
            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)
        else:
            print('save_path is None. Not saving anything')

    def load_from_txt(self, save_path=None):
        if not save_path==None:
            self.save_path = save_path
        
        if not os.path.exists(self.save_path):
            raise Exception(f"Can not found file in path {self.save_path}")

        with open(self.save_path, 'rb') as file:
            cav = pickle.load(file)
            self.concepts = cav.concepts
            self.bottleneck = cav.bottleneck
            self.hparams = cav.hparams
            self.accuracies = cav.accuracies
            self.cavs = cav.cavs
        
        
        


