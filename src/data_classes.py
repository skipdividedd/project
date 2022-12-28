import os
import re
import torch
from pathlib import Path
from typing import List, Tuple
from random import shuffle, seed
import neurox.data.extraction.transformers_extractor as transformers_extractor


def get_file_names(data_path) -> List[str]: 
    """"
    Найти, где лежат все .csv файлы
    data_path имеет вид: '/home/senya/Документы/project/data_en_gum'
    """
    
    files = []
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames: #обходим все файлы-категории
            file = os.path.join(dirname, filename)
            if not re.search(r'/.gitignore', file):
                files.append(file)
    return files


class ConvertSample:
    """"
    Gets .csv files, makes train & test split in .txt format, trying to balance data.
    """
    
    def __init__(self, path, train_size=2600, test_size=900, shuffle: bool = True) -> None: 

        self.shuffle = shuffle
        self.path = path
        self.project_path = str(Path(os.getcwd()).parents[0])
        self.category = re.search(r'[a-zA-Z]+(?=.csv)', path)[0]
        self.train_size = train_size
        self.test_size = test_size
        

    def read(self) -> List[str]: 

        seed(20)
        with open(self.path, encoding="utf-8") as f:
            lines = [line.split('\t') for line in f]
            
            if self.shuffle:
                shuffle(lines)
                
        return lines
    
    def stupid_cycle(self, values, dct, number) -> dict[str, int]: #util для семплинга
        
        dict_filter = {}
        
        for value in values:
            i = 0
            for k, v in dct.items():
                if v == value:
                    if i < number:
                        dict_filter[k] = v
                        i+=1
                
        return dict_filter
                
    def stupid_sampler(self) -> Tuple[dict, ...]: #семплинг данных
        
        sents = self.read()
        values_train = []
        values_test = []
        sents_train = []
        sents_test = []

        for line in sents:
            part, value, sentence = line[0], line[1], line[2]
            if 2 < len(sentence.split()) < 35:
                if part == 'tr':
                    values_train.append(value)
                    sents_train.append(sentence)
                    
                if part == 'te' or part== 'va':
                    values_test.append(value)
                    sents_test.append(sentence)
                    
        train_dict = dict(zip(sents_train, values_train))  
        test_dict = dict(zip(sents_test, values_test))

        A = set(values_train)
        B = set(values_test)
        values = A.intersection(B)
        
        number_one = round(self.train_size/len(values))
        number_two = round(self.test_size/len(values))

        dict_filter_train = self.stupid_cycle(values, train_dict, number_one)
        dict_filter_test = self.stupid_cycle(values, test_dict, number_two)
        
        return dict_filter_train, dict_filter_test

    

    def permute(self, dct) -> dict[str, int]: 
        # перемешивает словарь данных

        l = list(dct.items())
        shuffle(l)
        return dict(l)
        
    def using_shuffle(self, a) -> dict:

        keys = list(a.keys())
        values = list(a.values())
        shuffle(values)
        return dict(zip(keys, values))

    def create_dicts(self) -> Tuple[dict, ...]:
        
        dict_filter_train, dict_filter_test = self.stupid_sampler()

        if self.shuffle:
            dict_filter_train = self.permute(dict_filter_train)
            dict_filter_test = self.permute(dict_filter_test)
        
        dict_control_task = dict_filter_train.copy()
        dict_control_task = self.using_shuffle(dict_control_task)

        return dict_filter_train, dict_filter_test, dict_control_task

    def create_paths(self) -> Tuple[str, ...]:

        if re.search(r'(?<=\/)[a-zA-Z][a-zA-Z]_[a-zA-Z]+(?=_)', self.path)[0]:
            dataset = re.search(r'(?<=\/)[a-zA-Z][a-zA-Z]_[a-zA-Z]+(?=_)', self.path)[0]
            path = self.project_path+f'/data/large_data_{dataset}'
        else:
            path = self.project_path+'/data/large_data'
            
        if not os.path.isdir(path):
            os.mkdir(path)
            
        if not os.path.isdir(path+f'/data_{self.category}'):
            os.mkdir(path+f'/data_{self.category}')
        
        result_path_datatrain = path+f"/data_{self.category}/datatrain_{self.category}.txt"
        result_path_labeltrain = path+f"/data_{self.category}/labeltrain_{self.category}.txt"

        result_path_cdatatrain = path+f"/data_{self.category}/cdatatrain_{self.category}.txt"
        result_path_clabeltrain = path+f"/data_{self.category}/clabeltrain_{self.category}.txt"
        
        result_path_datatest = path+f"/data_{self.category}/datatest_{self.category}.txt"
        result_path_labeltest = path+f"/data_{self.category}/labeltest_{self.category}.txt"

        return result_path_datatrain, result_path_labeltrain, result_path_cdatatrain, result_path_clabeltrain, \
               result_path_datatest, result_path_labeltest


    def writer(self) -> Tuple[str, ...]: 
        
        """
        Writes to a file
        """
        result_datatrain, result_labeltrain, result_cdatatrain, result_clabeltrain, result_datatest, result_labeltest = self.create_paths()
       
        
        dict_filter_train, dict_filter_test, dict_control_task = self.create_dicts()

        with open(result_datatrain, "w", encoding="utf-8") as traindata, \
             open(result_labeltrain, "w", encoding="utf-8") as trainlabel, \
             open(result_cdatatrain, "w", encoding="utf-8") as ctraindata, \
             open(result_clabeltrain, "w", encoding="utf-8") as ctrainlabel, \
             open(result_datatest, "w", encoding="utf-8") as testdata, \
             open(result_labeltest, "w", encoding="utf-8") as testlabel:
            
    
            for sentence, value in dict_filter_train.items():
                traindata.writelines(sentence)
                trainlabel.writelines(value + '\n')

            for sentence, value in dict_control_task.items():
                ctraindata.writelines(sentence)
                ctrainlabel.writelines(value + '\n')

            for sentence, value in dict_filter_test.items():
                testdata.writelines(sentence)
                testlabel.writelines(value + '\n')
                                                                  
        
        return result_datatrain, result_labeltrain, result_cdatatrain, result_clabeltrain, result_datatest, result_labeltest
        


class GetEmbeddings:
    """"
    Receives .txt files with sentences and computes embeddings for them.
    """
    
    def __init__(self, path_trdata, path_tedata) -> None:
        
        self.path_trdata = path_trdata
        self.path_tedata = path_tedata
        self.project_path = str(Path(os.getcwd()).parents[0])
        self.category = re.search(r'(?<=_)[a-zA-Z]+(?=.txt)', path_trdata)[0]
        self.dataset = re.search(r'(?<=_)[a-zA-Z]+_[a-zA-Z]+(?=\/)', path_trdata)[0]
        self.path = self.project_path+f'/data/large_data_{self.dataset}/data_{self.category}'
        
    def jsons(self, model) -> None:
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()
        
        transformers_extractor.extract_representations(model,
        self.path_trdata,
        self.path+'/activations_train.json',
        device=device,
        aggregation="average" #last, first
        )
        
        transformers_extractor.extract_representations(model,
        self.path_tedata,
        self.path+'/activations_te.json',
        device=device,
        aggregation="average" #last, first
        )