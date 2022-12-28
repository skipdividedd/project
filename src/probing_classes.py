import os
import re
import time
import functools
import numpy as np
from typing import List
from pathlib import Path
import neurox.data.loader as data_loader
import neurox.interpretation.utils as utils
import neurox.interpretation.ablation as ablation
import neurox.interpretation.linear_probe as linear_probe
import neurox.data.extraction.transformers_extractor as transformers_extractor

def get_converted_filenames(data_path: str) -> list:
    """
    Получить пути к файлам, где уже посчитаны эмбеддинги.
    data_path вида: '/home/senya/Документы/project/large_data_en_gum'
    """

    directories = [] #тут делаем обход, чтобы получить имена каталогов
    for root,dirs,files in os.walk(data_path):
        for directory in dirs:
            directories.append(os.path.join(root, directory))

    all_paths = []
    for directory in directories: #тут делаем обход каталагов, чтобы получить имена файлов
        paths = []
        for (root,dirs,files) in os.walk(directory, topdown=True):
            for file in files:
                paths.append(os.path.join(root, file))
            all_paths.append(sorted(paths))

    number_cats = len(all_paths)
    check = [True]*number_cats
    # проверяем, что для всех категорий посчитаны эмбеддинги
    if [len(all_paths[i]) == 8 for i in range(number_cats)] == check:
        true_paths = [sorted(all_paths[i][4:]) for i in range(len(all_paths))]
        true_paths = [[i[1], i[3], i[0], i[2]] for i in true_paths]
        control_paths = [sorted(all_paths[i][2:5] + all_paths[i][6:7]) for i in range(len(all_paths))]
        #control_paths = [[i[1], i[3], i[0], i[2]] for i in control_paths]
        return true_paths, control_paths

    else:
        raise IndexError('Check if you got embeddings')


def load_sentence_data(source_path, labels_path, activations): #код не мой
    
    #тут немного переписали функцию потому что в библиотеке ошибка!!!

    tokens = {"source": [], "target": []}

    with open(source_path) as source_fp:
        for line_idx, line in enumerate(source_fp):
            line_tokens = line.strip().split() #вот тут переписано
            tokens["source"].append(line_tokens) #и тут

    with open(labels_path) as labels_fp:
        for line in labels_fp:
            line_tokens = line.strip().split()
            tokens["target"].append(line_tokens)

    assert len(tokens["source"]) == len(tokens["target"]), (
        "Number of lines do not match (source: %d, target: %d)!"
        % (len(tokens["source"]), len(tokens["target"]))
    )

    assert len(activations) == len(tokens["source"]), (
        "Number of lines do not match (activations: %d, source: %d)!"
        % (len(activations), len(tokens["source"]))
    )

    
    for idx, activation in enumerate(activations):
        assert activation.shape[0] == len(tokens["source"][idx])

    return tokens


class Experiment:
    """
    Class for probing.
    """
    
    def __init__(self, path_trdata: str, path_trlabel: str, path_tedata: str, path_telabel: str) -> None:
        
        self.category = re.search(r'(?<=_)[a-zA-Z]+(?=.txt)', path_trdata)[0] # grammar category
        self.dataset = re.search(r'(?<=_)[a-zA-Z]+_[a-zA-Z]+(?=\/)', path_trdata)[0] # dataset name
        self.path = str(Path(os.getcwd()).parents[0])+f'/data/large_data_{self.dataset}/data_{self.category}'
        
        # инициализированы переменные для пробинга
        
        self.activations_tr, self.num_layers = data_loader.load_activations(self.path+'/activations_train.json', 768)
        self.activations_te, self.num_layers = data_loader.load_activations(self.path+'/activations_te.json', 768)
        
        self.tokens_tr = load_sentence_data(path_trdata, path_trlabel, self.activations_tr)
        self.tokens_te = load_sentence_data(path_tedata, path_telabel, self.activations_te)
        
        self.X_tr, self.y_tr, self.mapping_tr = utils.create_tensors(self.tokens_tr, self.activations_tr, 'Nom')
        self.label2idx_tr, self.idx2label_tr, self.src2idx_tr, self.idx2src_tr = self.mapping_tr
        
        self.X_te, self.y_te, self.mapping_te = utils.create_tensors(self.tokens_te, self.activations_te, 'Nom')
        self.label2idx_te, self.idx2label_te, self.src2idx_te, self.idx2src_te = self.mapping_te
        
    def data_size(self) -> tuple:
        # returns len of train & test data and the number of classes
        return self.X_tr.shape[0], self.X_te.shape[0], len(set(self.y_te))

    def run_classification(self, metric: str = "accuracy"): 
        # log reg classification        
        probe = linear_probe.train_logistic_regression_probe(self.X_tr, self.y_tr, 
                                                             lambda_l1=0.001, lambda_l2=0.001)
        
        scores_tr = linear_probe.evaluate_probe(probe, self.X_tr, self.y_tr, 
                                                idx_to_class=self.idx2label_tr, metric=metric)
        
        scores_te = linear_probe.evaluate_probe(probe, self.X_te, self.y_te, 
                                                idx_to_class=self.idx2label_te, metric=metric)
        
        return probe, scores_tr, scores_te
    
    
    def layer_wise(self, n: int, metric: str ="accuracy"): 
        # for probing specific layers
        layer_n_X_tr = ablation.filter_activations_by_layers(self.X_tr, [n], 13)
        probe_layer_n = linear_probe.train_logistic_regression_probe(layer_n_X_tr, self.y_tr, 
                                                                     lambda_l1=0.001, lambda_l2=0.001)
        scores_tr = linear_probe.evaluate_probe(probe_layer_n, layer_n_X_tr, self.y_tr, 
                                                idx_to_class=self.idx2label_tr, metric=metric)
        
        layer_n_X_te = ablation.filter_activations_by_layers(self.X_te, [n], 13)
        scores_te = linear_probe.evaluate_probe(probe_layer_n, layer_n_X_te, self.y_te, 
                                                idx_to_class=self.idx2label_te, metric=metric)
        
        return probe_layer_n, scores_tr, scores_te
    
    
    def ranking(self, probe) -> np.ndarray:
        # neuron ranking
        ordering, cutoffs = linear_probe.get_neuron_ordering(probe, self.label2idx_tr)
    
        return ordering


class Ablation(Experiment):
    """
    Probing with ablation (zeroing out specific neurons).
    """

    def __init__(self, path_trdata: str, path_trlabel: str, path_tedata: str, path_telabel: str) -> None:
        super().__init__(path_trdata, path_trlabel, path_tedata, path_telabel)
    

    def keep_only(self, probe, n: int = 100): 
        # Here we train on top-n neurons.
        ordering, cutoffs = linear_probe.get_neuron_ordering(probe, self.label2idx_tr)
        X_tr_selected = ablation.filter_activations_keep_neurons(self.X_tr, ordering[:n])
        probe_selected = linear_probe.train_logistic_regression_probe(X_tr_selected, self.y_tr, lambda_l1=0.001, lambda_l2=0.001)
        scores_tr = linear_probe.evaluate_probe(probe_selected, X_tr_selected, self.y_tr, idx_to_class=self.idx2label_tr)
        X_te_selected = ablation.filter_activations_keep_neurons(self.X_te, ordering[:n])
        scores_te = linear_probe.evaluate_probe(probe_selected, X_te_selected, self.y_te, idx_to_class=self.idx2label_te)
        return ordering[:n], scores_tr, scores_te


    def remove_certain(self, probe, i: int = 9884):
        # Here we probe on bottom-n neurons.
        ordering, cutoffs = linear_probe.get_neuron_ordering(probe, self.label2idx_tr)
        X_tr_selected = ablation.filter_activations_remove_neurons(self.X_tr, ordering[:i])
        probe_selected = linear_probe.train_logistic_regression_probe(X_tr_selected, self.y_tr, lambda_l1=0.001, lambda_l2=0.001)
        scores_tr = linear_probe.evaluate_probe(probe_selected, X_tr_selected, self.y_tr, idx_to_class=self.idx2label_tr)
        X_te_selected = ablation.filter_activations_remove_neurons(self.X_te, ordering[:i])
        scores_te = linear_probe.evaluate_probe(probe_selected, X_te_selected, self.y_te, idx_to_class=self.idx2label_te)
        return ordering[i:], scores_tr, scores_te
        

def timer(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        print(f"{func.__name__} took {runtime:.4f} secs")
        return result
    return _wrapper


class Trainer:
    """
    Class which runs experiments for one or multiple categories. Has ablation (bool) flag. Attributes store scores.
    """
    
    def __init__(self, paths: list, ablation: bool = False) -> None:
        
        self.paths = paths
        self.ablation = ablation #if want to zero out neurons

        self.type = type(self.paths[0])


        self.data_size = {} 
        self.scores = {} #log reg scores
        self.ordered_neurons = {} #neuron ordering
        self.ablation_top_scores = {} #scores if keep top-n neurons
        self.ablation_bottom_scores = {} #scores if keep bottom-n neurons

        
    def ismanycats(func):
        @functools.wraps(func)
        def _wrapper(self, *args, **kwargs):
            
            if self.data_size:
                self.data_size.clear()

            if self.scores:
                self.scores.clear()

            if self.ordered_neurons:
                self.ordered_neurons.clear()

            if self.ablation_top_scores:
                self.ablation_top_scores.clear()

            if self.ablation_bottom_scores:
                self.ablation_bottom_scores.clear()
                

            if self.type is str:
                
                if not self.ablation:
                    cat = Experiment(self.paths[0], self.paths[1], self.paths[2], self.paths[3]) 
                else:
                    cat = Ablation(sself.paths[0], self.paths[1], self.paths[2], self.paths[3]) 
                self.data_size[cat.category] = [cat.data_size()]          
                func(self, cat, *args, **kwargs)            
            elif self.type is not str:
                for path in self.paths:
                    if not self.ablation:
                        cat = Experiment(path[0], path[1], path[2], path[3])  
                    else:
                        cat = Ablation(path[0], path[1], path[2], path[3])     
                    self.data_size[cat.category] = [cat.data_size()]  
                    func(self, cat, *args, **kwargs) 
        return _wrapper
    
    @timer
    @ismanycats
    def train_classification(self, cat, metric: str ='accuracy', n: int =100, i: int = 9884, goal: str = None):
        
        cat_name = cat.category
        probe, scores_tr, scores_te = cat.run_classification(metric=metric)
        self.scores[cat_name] = [scores_tr, scores_te] 
        
        if goal == 'ranking':
            neurons = cat.ranking(probe)
            self.ordered_neurons[cat_name] = neurons[:n]

        elif goal == 'keep_ablation':
            neurons, scores_tr, scores_te = cat.keep_only(probe, n=n)
            self.ordered_neurons[cat_name] = neurons
            self.ablation_top_scores[cat_name] = [scores_tr, scores_te] 

        elif goal == 'remove_ablation':
            neurons, scores_tr, scores_te = cat.remove_certain(probe, i=i)
            self.ordered_neurons[cat_name] = neurons
            self.ablation_bottom_scores[cat_name] = [scores_tr, scores_te]

              
    @timer
    @ismanycats
    def train_layers(self, cat, n: int = None, lrange: int = None, metric='accuracy'):
        
        cat_name = cat.category

        if n is None and range is None:

            for n in range(cat.num_layers):
                probe_layer_n, scores_tr, scores_te = cat.layer_wise(n, metric=metric)
                self.scores[cat_name, n] = [scores_tr, scores_te]

        elif n is not None:
            probe_layer_n, scores_tr, scores_te = cat.layer_wise(n, metric=metric)
            self.scores[cat_name, n] = [scores_tr, scores_te] 
            
        elif lrange is not None:
            for n in range(lrange):
                probe_layer_n, scores_tr, scores_te = cat.layer_wise(n, metric=metric)
                self.scores[cat_name, n] = [scores_tr, scores_te] 