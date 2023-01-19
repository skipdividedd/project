import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def return_dict(dct, lang):
    if lang == 'en' or lang == 'de':
        li = ['Definite', 'Case', 'Gender', 'Number', 'Degree', 'PronType', 'NumType', 'Person', 'VerbForm', 'Mood', 'Tense']
    elif lang == 'tr':
        li = ['Definite', 'Case', 'Number', 'Degree', 'PronType', 'NumType', 'Person', 'Personpsor', 'VerbForm', 'Mood', 'Voice', 'Aspect', 'Tense', 'Polarity']
    new_d = {}
    for i in li:
        new_d[i] = dct[i]
    return new_d



class Init:

    def __init__(self, path, lang, d_name):

        with open(f'{path}scores_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores = return_dict(pickle.load(f), lang)
    
        with open(f'{path}scores_c_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_c = return_dict(pickle.load(f), lang)
            
        # нейроны.......
        with open(f'{path}neurons_{lang}_{d_name}.pkl', 'rb') as f: # все с отсечками
            self.ordered_neurons = return_dict(pickle.load(f), lang)
            
        with open(f'{path}top_15_{lang}_{d_name}.pkl', 'rb') as f: #тут 10 проц
            self.top_neurons = return_dict(pickle.load(f), lang)
            
        with open(f'{path}bottom_15_{lang}_{d_name}.pkl', 'rb') as f: #тут 
            self.bottom_neurons = return_dict(pickle.load(f), lang)
            
        with open(f'{path}bottom_n2_{lang}_{d_name}.pkl', 'rb') as f: #тут 
            self.bottom_neurons2 = return_dict(pickle.load(f), lang)
            
        with open(f'{path}threshold_{lang}_{d_name}.pkl', 'rb') as f: # с "трешхолдом"
            self.ordered_neurons_thres = return_dict(pickle.load(f), lang)
            
        # тут с аблейшн.....
        with open(f'{path}scores_keep_bot_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_bot = return_dict(pickle.load(f), lang)
            
        with open(f'{path}scores_keep_bot2_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_bot2 = return_dict(pickle.load(f), lang)
            
        with open(f'{path}scores_keep_top_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_top = return_dict(pickle.load(f), lang)
            
        with open(f'{path}scores_keep_thres_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_thres = return_dict(pickle.load(f), lang)
            
        # для контрол таск 
        with open(f'{path}scores_keep_top_c_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_top_c = return_dict(pickle.load(f), lang)
            
        with open(f'{path}scores_keep_thres_c_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_thres_c = return_dict(pickle.load(f), lang)
            
        # сколько данных
        with open(f'{path}size_{lang}_{d_name}.pkl', 'rb') as f:
            self.size = return_dict(pickle.load(f), lang)


def bad_scores(scores):
    
    for k, v in scores.items():
        m1 = []
        m2 = []

        for i, j in v[0].items():

            if v[0][i] < 0.5:

                if not m1.__contains__(v[0]):
                    m1.append(v[0])

                print(k, i, 'train_score', v[0][i])
                
        if m1:
            print(k, 'train', m1)
            print('---------------------')

        for i, j in v[1].items():
            
            if v[1][i] < 0.5:

                if not m2.__contains__(v[1]):
                    m2.append(v[1])

                print(k, i, 'test_score', v[1][i])
                
        if m2:
            print(k, 'test', m2)
            print('---------------------')


def accuracy_plot(dct_acc, dct_data):
    cats=[k for k in dct_acc.keys()]
    assert [k for k in dct_acc.keys()] == [k for k in dct_data.keys()]
    accuracy_train=[round(v[0]['__OVERALL__'], 2) for k, v in dct_acc.items()]
    accuracy_test = [round(v[1]['__OVERALL__'], 2) for k, v in dct_acc.items()]
    train=[v[0] for k, v in dct_data.items()]
    test = [v[1] for k, v in dct_data.items()]
    num_classes = [v[2] for k, v in dct_data.items()]
    
    d = pd.DataFrame({'categories': cats, 'train_acc' : accuracy_train, 'test_acc': accuracy_test,
                      'train_data': train, 'test_data': test, 'num_classes': num_classes})    
    
    
    fig1 = px.bar(d, x='categories', y=['train_acc', 'test_acc'], template="plotly_white") 
    fig1.update_traces(texttemplate='%{y}', textposition='outside')
    fig2 = px.bar(d, x='categories', y=['train_data', 'test_data', 'num_classes'], template= "seaborn") 
    fig2.update_traces(texttemplate='%{y}', textposition='outside')
    fig3 = px.bar(d, x='categories', y='num_classes', template="plotly") 
    fig3.update_traces(texttemplate='%{y}', textposition='outside')
    fig3.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Accuracy', 'Data size', 'Num classes'))
    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.add_trace(fig1['data'][1], row=1, col=1)
    fig.add_trace(fig2['data'][0], row=2, col=1)
    fig.add_trace(fig2['data'][1], row=2, col=1)
    fig.add_trace(fig3['data'][0], row=3, col=1)
    fig.update_layout(height=1400, width=900)
    fig.update_layout(showlegend=False)
    
    fig.show()


def accuracy(dct_acc):
    cats=[k for k in dct_acc.keys()]
    accuracy_train=[round(v[0]['__OVERALL__'], 2) for k, v in dct_acc.items()]
    accuracy_test = [round(v[1]['__OVERALL__'], 2) for k, v in dct_acc.items()]
    d = pd.DataFrame({'categories': cats, 'train_acc' : accuracy_train, 'test_acc': accuracy_test})    
    fig = px.bar(d, x='categories', y=['train_acc', 'test_acc'], template="plotly_white", barmode='group') 
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    fig.show()


def check_selectivity(scores):
    m = []
    for k, v in scores.items():
        m.append(v[1]['__OVERALL__'])
    return np.array(m)


def get_len(dct):
    len_values = []
    for key, value in dct.items():
        len_values.append(len(value))
        
    d = pd.DataFrame({f'neurons for all {len(dct.keys())} categories': dct.keys(), 'neurons' : len_values})    
    fig = px.bar(d, x=f'neurons for all {len(dct.keys())} categories', y=['neurons'], template="plotly_white", barmode='group') 
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()



def common_neurons(d1, d2):
    common_cats = []
    for k1 in d1.keys():
        for k2 in d2.keys():   
            if k1 == k2:
                common_cats.append(k1)
            
    df = pd.DataFrame(columns=common_cats)
    df = df.fillna(0)
    for cat in common_cats:
        common_neurons = []
        p = set(d1[cat]) & set(d2[cat])
        common_neurons.append(len(p))
        df[cat] = common_neurons
    return df   


def common_neurons_percentage(d1, d2):
    common_cats = []
    for k1 in d1.keys():
        for k2 in d2.keys():   
            if k1 == k2:
                common_cats.append(k1)
            
    df = pd.DataFrame(columns=common_cats)
    df = df.fillna(0)
    for cat in common_cats:
        common_neurons = []
        p = len(set(d1[cat]) & set(d2[cat])) * 100 / len((set(d1[cat]) | set(d2[cat])))
        common_neurons.append(round(p, 2))
        df[cat] = common_neurons
    return df 


def get_bottom(d1, d2):
    d = {}
    for key, value in d1.items():
        for i, j in d2.items():
            if key == i:
                s = set(value[0])
                t = set(j)
                d[key] = s.difference(t)
    return d


def common_neurons_percentage_multiple(d1, d2, le=[5, 10, 20, 30, 40, 50, 75, 80, 90, 99]):
    common_cats = []
    for k1 in d1.keys():
        for k2 in d2.keys():   
            if k1 == k2:
                common_cats.append(k1)
    le_id = [str(i) + '%' for i in le]        
    df = pd.DataFrame(index=le_id, columns=common_cats)
    df = df.fillna(0)

    for cat in common_cats:
        common_neurons = []
        for l in le:
            p = len(set(d1[cat][0][:d1[cat][1][l-1]]) & set(d2[cat][0][:d2[cat][1][l-1]])) * 100 / len((set(d1[cat][0][:d1[cat][1][l-1]]) | set(d2[cat][0][:d2[cat][1][l-1]])))
            common_neurons.append(round(p, 2))
        df[cat] = common_neurons
    return df 


def common_neurons_multiple(d1, d2, le=[5, 10, 20, 30, 40, 50, 75, 80, 90, 99]):
    common_cats = []
    for k1 in d1.keys():
        for k2 in d2.keys():   
            if k1 == k2:
                common_cats.append(k1)
    le_id = [str(i) + '%' for i in le]        
    df = pd.DataFrame(index=le_id, columns=common_cats)
    df = df.fillna(0)

    for cat in common_cats:
        common_neurons = []
        for l in le:
            p = set(d1[cat][0][:d1[cat][1][l-1]]) & set(d2[cat][0][:d2[cat][1][l-1]])
            common_neurons.append(len(p))
        df[cat] = common_neurons
    return df 