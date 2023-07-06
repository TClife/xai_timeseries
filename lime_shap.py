import os
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sklearn 
import numpy as np 
import wandb
import logging 
import scipy as sp 
from sklearn.linear_model import Ridge, lars_path 
from sklearn.utils import check_random_state
import os
import os.path
import numpy as np
import pdb
from lime import explanation
from argparse import ArgumentParser
from collections import Counter
from math import comb 
from data import load_data
from models import VQ_Classifier
torch.set_num_threads(32) 
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Create args parser for labels and batch size 
parser = ArgumentParser() 
parser.add_argument('--labels', type=int, default=0)
parser.add_argument('--num_features', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--classification_model', type=str, default="/home/smjo/xai_timeseries/vqvae/saved_models/classification/mitbih/8/cnn_transformer.pt")
parser.add_argument('--vqvae_model', default = "/home/smjo/xai_timeseries/vqvae/saved_models/hard_mitbih/8/model_290.pt")
parser.add_argument('--task', type=str, default='xai', help="Task being done")
parser.add_argument('--dataset', type=str, default="mitbih")
parser.add_argument('--plot_dataset', type=bool, default=True)
parser.add_argument('--feature_selection', type=str, default='highest_weights')
parser.add_argument('--model_type', type=str, default="cnn_transformer")
parser.add_argument('--auc_classification', type=bool, default=False)
parser.add_argument('--quantizer', type=int, default=8)

args = parser.parse_args() 

ds = load_data(args.dataset, args.task)

# Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(ds))
val_size = int(0.1 * len(ds))
test_size = len(ds) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

#load classification args 
classification_model = torch.load(args.classification_model)
class_args = classification_model['args']
#ECG Dataset
net = VQ_Classifier(
    num_classes = class_args.num_classes,
    vqvae_model = class_args.vqvae_model,
    positions = class_args.positions,
    mask = class_args.mask,
    auc_classification = args.auc_classification,
    model_type = class_args.model_type
).to(device)

#load classification model
net.load_state_dict(classification_model['model_state_dict'])

#find masking regions 
masked_regions = net.mask_region(ds.data[:64,:].to(device))

#Explainer
def perturb_codebook(m, idx, channels, mask):
    for i in range(m.shape[2]):
        m[:,idx, i] = mask[i]
    return 

def perturb_random(m, idx, channels, unique_values):
    m[:, idx] = np.random.choice(unique_values)
    return

class TSDomainMapper(explanation.DomainMapper):
    def __init__(self, signal_names, num_slices, is_multivariate):
        """Init function.
        Args:
            signal_names: list of strings, names of signals
        """
        self.num_slices = num_slices
        self.signal_names = signal_names
        self.is_multivariate = is_multivariate
        
    def map_exp_ids(self, exp, **kwargs):
        # in case of univariate, don't change feature ids
        if not self.is_multivariate:
            return exp
        
        names = []
        for _id, weight in exp:
            # from feature idx, extract both the pair number of slice
            # and the signal perturbed
            nsignal = int(_id / self.num_slices)
            nslice = _id % self.num_slices
            signalname = self.signal_names[nsignal]
            featurename = "%d - %s" % (nslice, signalname)
            names.append((featurename, weight))
        return names

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):

        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):

        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, shapley_weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        # pdb.set_trace()
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        
        elif method == 'shap':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=shapley_weights) #Fit a ridge regression on the 5000 samples using the labels from the original model and weights from exponential kernel 

            train_score = clf.score(data, labels) #R^2 score of the model

            # Predict the values of the dependent variable in the training data
            y_pred = clf.predict(data)

            coef = clf.coef_ #coefficients of each of the 72 slices
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0] #Multiply the weight vectors to the original data 
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: x[1], #if abs(x[1]), then it will be abs(weights)
                    reverse=True) #Sort the weights and the index in which the abs(weights) are highest (positive or negative) 
                return np.array([x[0] for x in feature_weights[:num_features]])

        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights) #Fit a ridge regression on the 5000 samples using the labels from the original model and weights from exponential kernel 

            train_score = clf.score(data, labels) #R^2 score of the model

            # Predict the values of the dependent variable in the training data
            y_pred = clf.predict(data)

            coef = clf.coef_ #coefficients of each of the 72 slices
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0] #Multiply the weight vectors to the original data 
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: x[1], #if abs(x[1]), then it will be abs(weights)
                    reverse=True) #Sort the weights and the index in which the abs(weights) are highest (positive or negative) 
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 1:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights, shapley_weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   shapley_weight,
                                   label,
                                   num_features,
                                   feature_selection=None,
                                   model_regressor=None):

        weights = self.kernel_fn(distances) #Converts distances to proximity values (using exponential kernel)
        shapley_weights = (shapley_weight - min(shapley_weight)) / (max(shapley_weight) - min(shapley_weight))
        shapley_original = np.array([1]) #shapley value of original instance
        shapley_weights = np.concatenate((shapley_original, shapley_weights))
        labels_column = neighborhood_labels[:, label] #output probabilites for label 0 (neighborhood_labels = predictions for each label)
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               shapley_weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor #Now use the easy ridge regression model for the used features only 

        if feature_selection == "shap":
            easy_model.fit(neighborhood_data[:, used_features],
                        labels_column, sample_weight=shapley_weights)
            prediction_score = easy_model.score(
                neighborhood_data[:, used_features],
                labels_column, sample_weight=shapley_weights) #obtain the prediction score using just the ten features with highest weights
            
            print("R^2: ", prediction_score)

            local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        
        else:
            easy_model.fit(neighborhood_data[:, used_features],
                        labels_column, sample_weight=weights)
            prediction_score = easy_model.score(
                neighborhood_data[:, used_features],
                labels_column, sample_weight=weights) #obtain the prediction score using just the ten features with highest weights
            
            print("R^2: ", prediction_score)

            local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)

class LimeTimeSeriesExplainer(object):
    """Explains time series classifiers."""

    def __init__(self,
                 kernel_width=1,
                 verbose=False,
                 class_names=None,
                 feature_selection=None,
                 signal_names=["not specified"]
                 ):

        # exponential kernel
        def kernel(d): return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.base = LimeBase(kernel, verbose)
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.signal_names = signal_names

    def explain_instance(self,
                         img,
                         timeseries_instance, 
                         classifier_fn,
                         num_slices,
                         unique_values,
                         len_ts,
                         index,
                         num_index,
                         labels=None,
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         model_regressor=None,
                         replacement_method='mean',
                         mask = None):

        permutations, predictions, distances, shapley_weight = self.__data_labels_distances(
            img, timeseries_instance, classifier_fn, unique_values,
            num_samples, num_slices, index, num_index, len_ts, mask, replacement_method)

        is_multivariate = len(timeseries_instance.shape) > 1
        
        if self.class_names is None:
            self.class_names = [str(x) for x in range(predictions[0].shape[0])]

        domain_mapper = TSDomainMapper(self.signal_names, num_slices, is_multivariate)
        ret_exp = Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)
        ret_exp.predict_proba = predictions[0]

        if top_labels:
            labels = np.argsort(predictions[0])[-top_labels:]
            ret_exp.top_labels = list(predictions)
            ret_exp.top_labels.reverse()
        labels = [0,1]
        for label in labels:
            (ret_exp.intercept[int(label)],
             ret_exp.local_exp[int(label)],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                permutations, predictions,
                distances, shapley_weight, label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_labels_distances(cls,
                                img,
                                timeseries,
                                classifier_fn,
                                unique_values,
                                num_samples,
                                num_slices,
                                index,
                                num_index,
                                len_ts,
                                mask,
                                replacement_method='mean'):

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances( #[10000, 52]
                x, x[0].reshape([1, -1]), metric='cosine').ravel() * 100 #cosine similarity between perturbed samples and test sample

        def kernel_shap(x, M_num):
            shapley_weight = []
            M = M_num 
            for i in range(1, x.shape[0]):
                z = np.count_nonzero(x[i]) #Number of players
                weight = (M-1) / ((comb(M, z)) * z * (M-z)) #shapley values
                shapley_weight.append(weight)

            shap_weight = np.stack(shapley_weight)

            return shap_weight 

        num_channels = 1
        
        #deacts
        num_deacts = timeseries.shape[1]-1

        len_ts = len_ts
        values_per_slice = round(len_ts / num_slices) #6 = 286 / 50 
        deact_per_sample = np.random.randint(1, num_deacts + 1, num_samples - 1) #random integer between 0~48 with for 4999
        perturbation_matrix = np.ones((num_samples, num_channels, num_slices)) 
        features_range = range(num_slices) #range from 0 to 50 
        original_data = [timeseries.clone()]

        for i, num_inactive in enumerate(deact_per_sample, start=1): #each of the 5000 samples (perturbation for) 
            logging.info("sample %d, inactivating %d", i, num_inactive)
            # choose random slices indexes to deactivate
            inactive_idxs = np.random.choice(features_range, num_inactive,
                                             replace=False)
            num_channels_to_perturb = np.random.randint(1, num_channels+1) #1

            channels_to_perturb = np.random.choice(range(num_channels),
                                                   num_channels_to_perturb,
                                                   replace=False) #0
            
            logging.info("sample %d, perturbing signals %r", i,
                         channels_to_perturb)
            
            for chan in channels_to_perturb:
                perturbation_matrix[i, chan, inactive_idxs] = 0 #make zero for the positions that will randomly change and 1 for positions that stays the same
                
            tmp_series = timeseries.clone()

            for idx in inactive_idxs: #index of slices
                codebook_idx = idx
                start_idx = idx * values_per_slice
                end_idx = start_idx + values_per_slice
                end_idx = min(end_idx, len_ts)

                if replacement_method == "codebook":
                    #use 0th codebook indices as inactive 
                    perturb_codebook(tmp_series, codebook_idx, channels_to_perturb, mask)
                elif replacement_method == "random":
                    #use random codebook indice as inactive 
                    perturb_random(tmp_series, codebook_idx, channels_to_perturb, unique_values)
                    
            original_data.append(tmp_series) #changed all the positions 
        # predictions = classifier_fn(torch.stack(original_data))
        original_data = torch.stack(original_data)

        predictions, recons = net.predict(original_data)            
        
        # create a flat representation for features
        perturbation_matrix = perturbation_matrix.reshape((num_samples, num_channels * num_slices)) #[5000, 72]
        distances = distance_fn(perturbation_matrix) #Cosine similarity of each perturb samples with respect to time_series [5000,]
        shapley_weight = kernel_shap(perturbation_matrix, timeseries.shape[1])
        return perturbation_matrix, predictions, distances, shapley_weight

class Explanation(object):
    """Object returned by explainers."""

    def __init__(self,
                 domain_mapper,
                 mode='classification',
                 class_names=None,
                 random_state=None):

        self.random_state = 42
        self.mode = mode
        self.domain_mapper = domain_mapper
        self.local_exp = {}
        self.intercept = {}
        self.score = None
        self.local_pred = None
        if mode == 'classification':
            self.class_names = class_names
            self.top_labels = None
            self.predict_proba = None
        elif mode == 'regression':
            self.class_names = ['negative', 'positive']
            self.predicted_value = None
            self.min_value = 0.0
            self.max_value = 1.0
            self.dummy_label = 1


    def available_labels(self):
        """
        Returns the list of classification labels for which we have any explanations.
        """
        try:
            assert self.mode == "classification"
        except AssertionError:
            raise NotImplementedError('Not supported for regression explanations.')
        else:
            ans = self.top_labels if self.top_labels else self.local_exp.keys()
            return list(ans)

    def as_list(self, label=None, **kwargs):

        label_to_use = label if self.mode == "classification" else self.dummy_label
        ans = self.local_exp[label_to_use]
        ans = [(x[0], float(x[1])) for x in ans]
        return ans

    def as_map(self):
        return self.local_exp

    def as_pyplot_figure(self, label=None, **kwargs):

        import matplotlib.pyplot as plt
        exp = self.as_list(label=label, **kwargs)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        # plt.yticks(pos, names)
        if self.mode == "classification":
            title = 'Local explanation for class %s' % self.class_names[label]
        else:
            title = 'Local explanation'
        plt.title(title)
        plt.ylim(-5,5)
        return fig

unique_values = range(0, 64)

#params
for param in net.parameters():
    param.requires_grad = False 

test_sample_codebooks = []
label_codebooks = []
label_weights = []
label_position = []
for k, (data, labels) in enumerate(test_loader):
    labels = torch.argmax(labels, axis=1)
    
    #codebook
    net.eval()
    net = net.to(device)
    data = data.unsqueeze(0).to(device)
    y_hat, codebook, data_recon,_ = net(data) #y_hat is logits, codebook is the codebook values, and emb is the embedded codebook values
    # Explain ECG Dataset
    num_features = args.num_features
    num_slices = codebook.shape[1]
    len_ts = data.shape[2]
    
    #Number of perturb indices
    index = list(range(num_slices))    
    num_indices = len(index)

    explainer = LimeTimeSeriesExplainer(class_names =['Class0', 'Class1'], feature_selection = args.feature_selection)
    exp = explainer.explain_instance(data, codebook, net(data), num_features = num_features, unique_values = unique_values, len_ts = len_ts, index=index, num_index=num_slices, labels=int(labels), num_samples = 5000, num_slices = num_slices, replacement_method="codebook", mask = masked_regions)
    exp.as_pyplot_figure(label=int(labels))

    values_per_slice = round(len_ts / num_slices)
    all_codebooks = []
    all_weights = []
    all_feature = []
    for i in range(num_features):
        feature, weight = exp.as_list(int(labels))[i]
        start = feature * values_per_slice
        end = start + values_per_slice
        color = 'red' if weight < 0 else 'green'
        if weight > 0: 
            all_weights.append(weight)
        if abs(weight) > 1:
            if weight < 0:
                weight = -1
            else:
                weight = 1
        all_codebooks.append(codebook[0,:, 0][feature])
        if abs(weight*1000000) > 1:
            plot_weight = 1
        else:
            plot_weight = weight
        plt.axvspan(start , end, color=color, alpha=abs(plot_weight))
        all_feature.append(feature)

    print("codebooks: ",all_codebooks)
    print("weights: ", all_weights)

    all_codebooks = [int(i) for i in all_codebooks]

    #save all_codebooks and all_weights and codes to text file
    test_sample_codebooks.append(codebook)

    label_codebooks.append(all_codebooks)
    label_weights.append(all_weights)
    label_position.append(all_feature)

    plt.plot(data.squeeze(1).squeeze(0).cpu().detach().numpy(), color = 'b', label="Original")
    plt.plot(data_recon.squeeze(1).squeeze(0).cpu().detach().numpy(), color = 'r', label="Reconstruction")
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    
    os.makedirs("./ecg_sample_{}_{}_{}_{}".format(args.dataset, args.feature_selection, args.model_type, args.quantizer), exist_ok=True)
    os.makedirs("./ecg_sample_{}_{}_{}_{}/label{}".format(args.dataset, args.feature_selection, args.model_type, args.quantizer, int(labels)), exist_ok=True)
    plt.savefig("./ecg_sample_{}_{}_{}_{}/label{}/sample{}".format(args.dataset, args.feature_selection, args.model_type, args.quantizer, int(labels),k))
    plt.show()
    plt.clf()
    
label_save = {
    "label": args.labels,
    "codebooks": label_codebooks,
    "weights": label_weights,
    "positions": label_position,
    "codes": test_sample_codebooks
} 

torch.save(label_save, "./ecg_sample_{}_{}_{}_{}/label{}/model_type_{}_codebooks_weights.pt".format(args.dataset, args.feature_selection, args.model_type, args.quantizer, args.labels, args.model_type))   

#positional rankings 
ptb_lime_quantizer1 = torch.load("./ecg_sample_{}_{}_{}_{}/label{}/model_type_{}_codebooks_weights.pt".format(args.dataset, args.feature_selection, args.model_type, args.quantizer, args.labels, args.model_type))
top_indices = [int(item) for sublist in ptb_lime_quantizer1['positions'] for item in sublist]
count = Counter(top_indices).most_common()
print("./ecg_sample_{}_{}_{}_{}/label{}/model_type_{}_codebooks_weights.pt".format(args.dataset, args.feature_selection, args.model_type, args.quantizer, args.labels, args.model_type))
print("Positional Rankings: ", count)