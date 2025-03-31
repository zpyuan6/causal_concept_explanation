import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torch import Tensor
import random

class CausalConceptVAE(nn.Module):

    def __init__(self,
                 input_shape:list = None,
                 concept_dims: list = [10,10,10,10],
                 kld_weight: int = 1,
                 classify_weight: int = 1,
                 model_file_path: str = None,
                 concept_remove_y: bool = False,
                 **kwargs) -> None:
        super(CausalConceptVAE, self).__init__()

        if model_file_path is not None:
            input_shape, concept_dims, concept_remove_y = self.load_from_pth(model_file_path)
        
        self.concept_dims = concept_dims
        if concept_remove_y:
            self.concept_dims = self.concept_dims[:-1]
        self.concept_remove_y = concept_remove_y
        self.kld_weight = kld_weight

        # input_shape = [neuronal abstraction width, number of layers]
        self.input_shape = input_shape

        self.classify_weight = classify_weight
        num_concepts = len(self.concept_dims)

        # Build Encoder
        self.encoder_layer_wise_layer = nn.Sequential(
                    # nn.Linear(self.input_shape[1], self.input_shape[1]),
                    # nn.LeakyReLU(),
                    nn.Linear(self.input_shape[1], num_concepts),
                    nn.LeakyReLU()
                )

        self.encoder_grouped_linear_feature_layer = nn.ModuleList([
            nn.Sequential(
                    nn.Linear(
                        self.input_shape[0], 
                        int((self.input_shape[0]+concept_dim)/2)
                        ),
                    nn.BatchNorm1d(
                        int((self.input_shape[0]+concept_dim)/2)
                        ),
                    nn.LeakyReLU(),
                    nn.Linear(
                        int((self.input_shape[0]+concept_dim)/2), 
                        int(self.input_shape[0]/4+3*concept_dim/4)
                        ),
                    nn.BatchNorm1d(int(self.input_shape[0]/4+3*concept_dim/4)),
                    nn.LeakyReLU(),
                    nn.Linear(
                        int(self.input_shape[0]/4+3*concept_dim/4), 
                        int(self.input_shape[0]/4+3*concept_dim/4)
                        ),
                    nn.BatchNorm1d(int(self.input_shape[0]/4+3*concept_dim/4)),
                    nn.LeakyReLU()
                ) for concept_dim in self.concept_dims])

        self.encoder_grouped_linear_concept_classifer = nn.ModuleList([
            nn.Sequential(
                    nn.Linear(int(self.input_shape[0]/4+3*concept_dim/4), concept_dim),
                    nn.BatchNorm1d(concept_dim),
                    nn.LeakyReLU()
                ) for concept_dim in self.concept_dims])

        self.encoder_grouped_fc_mu = nn.ModuleList([        
            nn.Sequential(
                    nn.Linear(int(self.input_shape[0]/4+3*concept_dim/4), concept_dim),
                    nn.LeakyReLU()
                ) for concept_dim in self.concept_dims])

        self.encoder_grouped_fc_var = nn.ModuleList([
            nn.Sequential(
                    nn.Linear(int(self.input_shape[0]/4+3*concept_dim/4), concept_dim),
                    nn.LeakyReLU()
                ) for concept_dim in self.concept_dims])

        if model_file_path:
            mask = torch.load(model_file_path)['causal_adjacency_mask'].cpu()
            print(f'Load causal_adjacency_mask: \n', mask)
        else:
            if self.concept_remove_y:
                mask = torch.ones(len(self.concept_dims)+1, len(self.concept_dims)+1) - torch.eye(len(self.concept_dims)+1)
            else:
                mask = (torch.ones(len(self.concept_dims), len(self.concept_dims)) - torch.eye(len(self.concept_dims)))
        
        self.causal_adjacency_mask = nn.Parameter(mask, requires_grad=False)

        # Build Decoder
        self.decoder_grouped_linear_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                        2*concept_dim,
                        2*concept_dim
                        ),
                nn.BatchNorm1d(2*concept_dim),
                nn.LeakyReLU(),

                    nn.Linear(
                        2*concept_dim,
                        int(self.input_shape[0]/4+3*concept_dim/2)
                        ),
                    nn.BatchNorm1d(int(self.input_shape[0]/4+3*concept_dim/2)),
                    nn.LeakyReLU(),

                    nn.Linear(
                        int(self.input_shape[0]/4+3*concept_dim/2),
                        int(self.input_shape[0]/4+3*concept_dim/2)
                        ),
                    nn.BatchNorm1d(int(self.input_shape[0]/4+3*concept_dim/2)),
                    nn.LeakyReLU(),

                    nn.Linear(
                        int(self.input_shape[0]/4+3*concept_dim/2),
                        int((self.input_shape[0]/2+concept_dim))
                        ),
                    nn.BatchNorm1d(int((self.input_shape[0]/2+concept_dim))),
                    nn.LeakyReLU(),

                    nn.Linear(
                        int(self.input_shape[0]/2+concept_dim),
                        int(self.input_shape[0]/2+concept_dim)
                        ),
                    nn.BatchNorm1d(int(self.input_shape[0]/2+concept_dim)),
                    nn.LeakyReLU(),

                    nn.Linear(
                        int((self.input_shape[0]/2+concept_dim)),
                        self.input_shape[0]
                        ),
                    nn.BatchNorm1d(self.input_shape[0]),
                    nn.LeakyReLU()
                ) for concept_dim in self.concept_dims])

        if concept_remove_y:
            self.decoder_layer_wise_layer = nn.Sequential(
                    nn.Linear(num_concepts+1, num_concepts+1),
                    nn.LeakyReLU(),
                    nn.Linear(num_concepts+1, self.input_shape[1]),
                    nn.LeakyReLU(),
                    nn.Linear(self.input_shape[1], self.input_shape[1]),
                    nn.Tanh()
                )
        else:
            self.decoder_layer_wise_layer = nn.Sequential(
                    nn.Linear(num_concepts, num_concepts),
                    nn.LeakyReLU(),
                    nn.Linear(num_concepts, self.input_shape[1]),
                    nn.LeakyReLU(),
                    nn.Linear(self.input_shape[1], self.input_shape[1]),
                    nn.Tanh()
                )

        if model_file_path is not None:
            self.load_state_dict(torch.load(model_file_path))


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_layer_wise_layer(input)
        
        concept_classes = []
        mu = []
        log_var = []
        for i, concept_linear in enumerate(self.encoder_grouped_linear_feature_layer):
            concept_feature = result[:,:,i]
            concept_latent = concept_linear(concept_feature)
            concept_class = self.encoder_grouped_linear_concept_classifer[i](concept_latent)
            one_mu = self.encoder_grouped_fc_mu[i](concept_latent)
            one_log_var = self.encoder_grouped_fc_var[i](concept_latent)
            concept_classes.append(concept_class)
            mu.append(one_mu)
            log_var.append(one_log_var)

        # mu = torch.cat(mu, dim = -1) # [B x num_concept]
        # log_var = torch.cat(log_var, dim = -1) # [B x num_concept]

        return [mu, log_var, concept_classes]

    def decode(self, z: Tensor, concept_classes: list, input_tensor: Tensor=None) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x num_concept]
        :param concept_classes: (Tensor) [num_concept x B x concept_dim]
        :return: (Tensor) [B x C x H x W]
        """
        concept_features = []
        for index, decoder_linear in enumerate(self.decoder_grouped_linear_layer):
            latent_input = torch.cat([concept_classes[index], z[index]], dim = -1) 
            concept_feature = decoder_linear(latent_input)
            concept_features.append(concept_feature)
        
        num_concept = len(concept_features)
        if self.concept_remove_y:
            concept_features.append(input_tensor[:,:, -1])
            num_concept += 1

        if self.training:
            mask_index = random.randint(0, num_concept-1)
            concept_features[mask_index] = concept_features[mask_index] * 0.0

        concept_features = torch.stack(concept_features, dim = -1)

        result = self.decoder_layer_wise_layer(concept_features)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, concept_classes = self.encode(input)
        z = [self.reparameterize(mu[i], log_var[i]) for i in range(len(self.concept_dims))]
        concept_classes = [F.softmax(concept_class, dim = -1) for concept_class in concept_classes]
        return  [self.decode(z, concept_classes, input), input, mu, log_var, concept_classes]

    def loss_function(
                    self,
                    *args,
                    **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        concept_classes = args[4]

        recons_loss =F.mse_loss(recons, input, reduction='sum') / input.shape[0] / input.shape[1] / input.shape[2]

        classify_loss = 0

        if len(args) > 5:
            concept_labels = args[5]
            classify_loss_function = nn.NLLLoss()
            classify_loss = sum([
                torch.mean(
                    classify_loss_function(
                        torch.log(concept_class),
                        concept_labels[:,index]
                        ), 
                    dim=0) 
                        for index, concept_class in enumerate(concept_classes)
                        ]) 
            # / len(concept_classes)
        else:
            classify_loss = 0
        
        kld_loss = sum([torch.mean(-0.5 * torch.sum(1 + log_var[i] - mu[i]** 2 - log_var[i].exp(), dim = 1), dim = 0) for i in range(len(self.concept_dims))]) / len(self.concept_dims)
 
        loss = recons_loss + self.kld_weight * kld_loss + self.classify_weight * classify_loss
        return {'loss': loss, 'Reconstruction_Loss': self.kld_weight * recons_loss.detach(), 'KLD': kld_loss.detach(), 'Classify_Loss': self.classify_weight * classify_loss.detach()}

    # def sample(
    #     self,
    #     z:  Tensor,
    #     concept_one_hot: Tensor,
    #     current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = z.to(current_device)
    #     concept_one_hot = [one_hot.to(current_device) for one_hot in concept_one_hot]
    #     samples = self.decode(z, concept_one_hot)
    #     return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:

        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_hyperparamters(self):
        return {
            'input_shape': self.input_shape,
            'kld_weight': self.kld_weight,
            'concept_dims': self.concept_dims,
            'classify_weight': self.classify_weight
        }


# <----------------- Concept Representations ------------------------>
    def get_concept_representations(self) -> List[Tensor]:
        
        # encoder_connectivity = (self.encoder_layer_wise_layer[0].weight**2).T @ (self.encoder_layer_wise_layer[2].weight**2).T
        encoder_connectivity = (self.encoder_layer_wise_layer[0].weight**2).T
        concept_locations = torch.argmax(encoder_connectivity, dim=0)

        concept_representations_for_each_concept_value = []
        concept_representations_for_each_concept= []

        for i in range(concept_locations.shape[0]):
            concept_connectivity=(self.encoder_grouped_linear_feature_layer[i][0].weight**2).T @ (self.encoder_grouped_linear_feature_layer[i][3].weight**2).T @ (self.encoder_grouped_linear_feature_layer[i][6].weight**2).T @ (self.encoder_grouped_linear_concept_classifer[i][0].weight**2).T

            concept_representations_for_each_concept_value.append(concept_connectivity)
            concept_representations_for_each_concept.append(torch.sum(concept_connectivity, dim=0))

        if self.concept_remove_y:
            concept_locations = torch.cat([concept_locations, torch.tensor([self.input_shape[1]-1]).to(concept_locations.device)], dim=0)

        return concept_locations, concept_representations_for_each_concept, concept_representations_for_each_concept_value





# <---------------- Causal Structure Discovery ------------------------>

    def get_causal_adjacency_matrix(self):

        # max_indices = torch.argmax(self.encoder_layer_wise_layer[2].weight**2, dim=1)
        # encoder_connectivity = (self.encoder_layer_wise_layer[0].weight**2).T @ (self.encoder_layer_wise_layer[2].weight**2).T
        encoder_connectivity = (self.encoder_layer_wise_layer[0].weight**2).T
        max_indices = torch.argmax(encoder_connectivity, dim=0)

        if self.concept_remove_y:
            max_indices = torch.cat([max_indices, torch.tensor([self.input_shape[1]-1]).to(max_indices.device)], dim=0)

        # # adjacency_matrix is defined as the product of the weights of the layer_wise_layer in decoder

        connectivity =  (self.decoder_layer_wise_layer[0].weight**2).T @ (self.decoder_layer_wise_layer[2].weight**2).T @ (self.decoder_layer_wise_layer[4].weight**2).T
        adjacency_matrix = connectivity[:, max_indices]

        # # adjacency_matrix is defined as the product of the weights of the layer_wise_layer in decoder
        # connectivity = ((self.encoder_layer_wise_layer[0].weight**2).T[max_indices, :]) @ (self.encoder_layer_wise_layer[2].weight**2).T @ (self.decoder_layer_wise_layer[0].weight**2).T @ ((self.decoder_layer_wise_layer[2].weight**2).T[:, max_indices])
        # adjacency_matrix = connectivity

        mask = adjacency_matrix > 1e-4

        self.causal_adjacency_mask = nn.Parameter((self.causal_adjacency_mask.bool() & mask).float(), requires_grad=False)
        
        adjacency_matrix = adjacency_matrix * self.causal_adjacency_mask

        return adjacency_matrix

    def h_loss(self, adjacency_matrix: Tensor):
        return torch.trace(torch.matrix_exp(adjacency_matrix * adjacency_matrix)) - adjacency_matrix.shape[0]


# <---------------- Load Module ------------------------>

    def load_from_pth(self, path):
        parameter = torch.load(path)

        layer_num = parameter['encoder_layer_wise_layer.0.weight'].shape[1]
        concept_num = parameter['encoder_layer_wise_layer.0.weight'].shape[0]

        input_witdth = parameter['encoder_grouped_linear_feature_layer.0.0.weight'].shape[1]

        input_shape = [input_witdth, layer_num]
        concept_dims = []

        for concept_index in range(concept_num):
            concept_dims.append(parameter[f'encoder_grouped_linear_concept_classifer.{concept_index}.0.weight'].shape[0])

        concept_remove_y = (concept_num != parameter['decoder_layer_wise_layer.0.weight'].shape[0])
        if concept_remove_y:
            concept_dims.append(parameter['encoder_grouped_linear_concept_classifer.0.0.weight'].shape[0])

        return input_shape, concept_dims, concept_remove_y

