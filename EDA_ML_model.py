#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# define the EDA class

class crop_recommendation_visulization():
    
    def __init__(self, data_file):
        self.data=pd.read_csv(data_file)
    
    def visulize_factors(self):
        df = self.data
        fig, axes  = plt.subplots(7, 3, figsize=(20,15), gridspec_kw=dict(hspace=0.5, wspace=0.1))
        colors= ['red', 'orange', 'blue', 'pink', 'violet', 'lightgreen', 'grey']
        for i, j in enumerate(df.columns[:-1]):
            sns.histplot(data=df, x=j, ax=axes[i,0], kde=True, color=colors[i])
            sns.violinplot(data=df, x=j, ax=axes[i,1], color=colors[i])
            sns.boxplot(data=df, x=j, ax=axes[i,2], color=colors[i])
            
    def comparison_of_attributes(self):
        df = self.data
        fig, axes  = plt.subplots(7, 1, figsize=(25,45), gridspec_kw=dict(hspace=0.2))
        for i, j in enumerate(df.columns[:-1]):
            sns.barplot(data=df, x='label', y=j, capsize=0.1, ax=axes[i])
            axes[i].set_xlabel(' ')
        for a in fig.axes:
            for c in a.containers:
                a.bar_label(c, padding=18)
            a.spines[['top', 'right']].set_visible(False)
    
    def top_5_most_requried_crop(self):
        df=self.data
        fig = plt.figure(figsize=(30,25))
        grid = plt.GridSpec(4,4, hspace=0.3)
        sns.barplot(data = df.groupby('label').mean()['N'].sort_values(ascending=False)[:5].reset_index(), x='label', y='N', ax=fig.add_subplot(grid[1,1]), alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['P'].sort_values(ascending=False)[:5].reset_index(), x='label', y='P', ax=fig.add_subplot(grid[1,2]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['K'].sort_values(ascending=False)[:5].reset_index(), x='label', y='K', ax=fig.add_subplot(grid[1,3]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['temperature'].sort_values(ascending=False)[:5].reset_index(), x='label', y='temperature', ax=fig.add_subplot(grid[2,1]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['humidity'].sort_values(ascending=False)[:5].reset_index(), x='label', y='humidity', ax=fig.add_subplot(grid[2,2]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['ph'].sort_values(ascending=False)[:5].reset_index(), x='label', y='ph', ax=fig.add_subplot(grid[2,3]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['rainfall'].sort_values(ascending=False)[:5].reset_index(), x='label', y='rainfall',ax=fig.add_subplot(grid[3,1:]),alpha=0.7, edgecolor='black')
        for i in fig.get_axes():
            i.set_xlabel('')
            i.spines[['top', 'right']].set_visible(False)
            for j in i.containers:
                i.bar_label(j, padding=5)
            i.set_title(f'Top 5 most {i.get_ylabel()} required crops\n', font='monospace', weight='semibold', size=12)
            
    def top_5_least_requried_crop(self):
        df=self.data
        fig = plt.figure(figsize=(30,25))
        grid = plt.GridSpec(4,4, hspace=0.3)
        sns.barplot(data = df.groupby('label').mean()['N'].sort_values(ascending=False)[-5:].reset_index(), x='label', y='N', ax=fig.add_subplot(grid[1,1]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['P'].sort_values(ascending=False)[-5:].reset_index(), x='label', y='P', ax=fig.add_subplot(grid[1,2]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['K'].sort_values(ascending=False)[-5:].reset_index(), x='label', y='K', ax=fig.add_subplot(grid[1,3]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['temperature'].sort_values(ascending=False)[-5:].reset_index(), x='label', y='temperature', ax=fig.add_subplot(grid[2,1]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['humidity'].sort_values(ascending=False)[-5:].reset_index(), x='label', y='humidity', ax=fig.add_subplot(grid[2,2]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['ph'].sort_values(ascending=False)[-5:].reset_index(), x='label', y='ph', ax=fig.add_subplot(grid[2,3]),alpha=0.7, edgecolor='black')
        sns.barplot(data = df.groupby('label').mean()['rainfall'].sort_values(ascending=False)[-5:].reset_index(), x='label', y='rainfall',ax=fig.add_subplot(grid[3,1:]),alpha=0.7, edgecolor='black')
        for i in fig.get_axes():
            i.set_xlabel('')
            i.spines[['top', 'right']].set_visible(False)
            for j in i.containers:
                i.bar_label(j, padding=5)
            i.set_title(f'Top 5 least {i.get_ylabel()} required crops\n', font='monospace', weight='semibold', size=12)
            
    def Two_component_visulization(self):
        df = self.data
        pca_2 = PCA(n_components=2)
        PCA_2 = pca_2.fit_transform(df.drop('label', axis=1))
        PCA_df = pd.DataFrame(PCA_2)
        PCA_df.columns= ['component_1', 'component_2']
        fig = px.scatter(PCA_df,x='component_1', y='component_2', color=df['label'] , labels={'color':'crop'}, title=f'Decomposed data with a {100*pca_2.explained_variance_ratio_.sum():1.0f}% explained variance')
        fig.show()
        
    def Three_component_visulization(self):
        df=self.data
        pca_3 = PCA(n_components=3)
        PCA_3 = pca_3.fit_transform(df.drop('label', axis=1))
        PCA_df = pd.DataFrame(PCA_3)
        PCA_df.columns= ['component_1', 'component_2', 'component_3']
        fig = px.scatter_3d(PCA_df,x='component_1', y='component_2', z='component_3',color=df['label'] , labels={'color':'crop'}, title=f'Decomposed data with a {100*pca_3.explained_variance_ratio_.sum():1.0f}% explained variance')
        fig.show()
        
    def corr_plot(self):
        df = self.data
        sns.heatmap(df.corr(), cbar=False, annot=True, cmap='viridis', linewidth=0.1,alpha=0.8)
        fig = plt.gcf()
        fig.set_size_inches(10,6)
        plt.xticks(rotation=360);
        
    def pairplot(self):
        df=self.data
        sns.pairplot(df, hue='label')
        
        
class crop_recommendation_model():
    
    def __init__(self, model_file, encoder):
        with open(model_file, 'rb') as model_file, open(encoder, 'rb') as encoder:
            self.model = pickle.load(model_file)
            self.encoder = pickle.load(encoder)
            self.data = None
        
    
    def predict_proba(self, data_file):
        
        df = pd.read_csv(data_file)
        X = df.drop('label', axis=1).copy()
        pred_prob = self.model.predict_proba(X)
        pred_prob = pd.DataFrame(self.model.predict_proba(X))
        pred_prob.columns=self.encoder.classes_
        return pred_prob
    
        
    def predicted_output(self, data_file):
        
        df = pd.read_csv(data_file)
        X = df.drop('label', axis=1).copy()
        X['prediction'] = self.encoder.classes_[np.argmax(self.model.predict_proba(X), axis=1)]
        return X
        
    


# In[ ]:





# In[ ]:




