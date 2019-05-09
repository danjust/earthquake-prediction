import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def get_random(decrease,increase):
    r = np.random.rand()
    if r<decrease:
        return -1
    elif r>1-increase:
        return 1
    else:
        return 0


class model():
    def __init__(
            self,
            featureset,
            layerlist,
            dropoutlist,
            num_all_features,
            mutationrate_features = .05,
            mutationrate_layers = .05,
            mutationrate_layersize = .05,
            std_layersize = 8,
            mutationrate_dropout = .05,
            std_dropoutrate = .1):

        self.featureset = featureset
        self.layerlist = layerlist
        self.dropoutlist = dropoutlist
        self.num_all_features = num_all_features

        self.mutationrate_features = mutationrate_features
        self.mutationrate_layers = mutationrate_layers
        self.mutationrate_layersize = mutationrate_layersize
        self.std_layersize = std_layersize
        self.mutationrate_dropout = mutationrate_dropout
        self.std_dropoutrate = std_dropoutrate

        self.num_features = len(featureset)
        self.num_layers = len(layerlist)
        self.available_features = set(range(num_all_features))-featureset

    def mutate(self):
        # add or remove one feature
        change_num_features = get_random(
                self.mutationrate_features/2,
                self.mutationrate_features/2
        )
        if change_num_features==1:
            new_feature = np.random.choice(list(self.available_features))
            self.featureset.add(new_feature)
            self.num_features = self.num_features+1
            self.available_features = self.available_features-self.featureset
        if change_num_features==-1:
            if self.num_features>1:
                remove_feature = np.random.choice(list(featureset))
                self.featureset = self.featureset - set(remove_feature)
                self.num_features = self.num_features-1
                self.available_features.add(remove_feature)

        # add or remove one layer
        change_num_layers = get_random(
                self.mutationrate_layers/2,
                self.mutationrate_layers/2
        )
        if change_num_layers==1:
            self.layerlist.append(self.layerlist[self.num_layers-1])
            self.dropoutlist.append(self.dropoutlist[self.num_layers-1])
            self.num_layers = self.num_layers+1
        if change_num_layers==-1:
            if self.num_features>1:
                self.layerlist.pop(self.num_layers-1)
                self.dropoutlist.pop(self.num_layers-1)
                self.num_layers = self.num_layers-1

        # change units per layer
        for i in range(self.num_layers):
            if np.random.rand()<self.mutationrate_layersize:
                self.layerlist[i] = int(np.round(np.random.normal(
                        self.layerlist[i],
                        self.std_layersize
                )))
                if self.layerlist[i]<1:
                    self.layerlist[i]=1

        # change dropoutrate
        for i in range(self.num_layers):
            if np.random.rand()<self.mutationrate_dropout:
                self.dropoutlist[i] = int(np.round(np.random.normal(
                        self.dropoutlist[i],
                        self.std_dropoutrate
                )))
                if self.dropoutlist[i]<0:
                    self.dropoutlist[i]=0
                if self.dropoutlist[i]>0.5:
                    self.dropoutlist[i]=0.5


    def build(self):
        """Function to create the model architecture"""
        model = Sequential()
        if self.num_layers==0:
            model.add(Dense(
                    units=1,
                    input_dim=self.num_features,
                    activation='relu',
                    use_bias=False,
            ))
        else:
            model.add(Dense(
                    units=self.layerlist[0],
                    input_dim=self.num_features,
                    activation='relu',
            ))
            if self.dropoutlist[0]>0:
                model.add(Dropout(rate=self.dropoutlist[0]))

            for i in range(1,self.num_layers):
                model.add(Dense(
                        units=self.layerlist[i],
                        activation='relu',
                ))
                if self.dropoutlist[i]>0:
                    model.add(Dropout(rate=self.dropoutlist[i]))

            model.add(Dense(
                    units=1,
                    activation='relu',
                    use_bias=False,
            ))
        return model
