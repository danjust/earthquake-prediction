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
            reg_constant,
            learning_rate,
            lr_decay,
            num_all_features,
            std_layersize = 8,
            std_dropoutrate = .1):

        self.featureset = featureset
        self.layerlist = layerlist
        self.dropoutlist = dropoutlist
        self.reg_constant = reg_constant
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.num_all_features = num_all_features

        self.std_layersize = std_layersize
        self.std_dropoutrate = std_dropoutrate

        self.num_features = len(featureset)
        self.num_layers = len(layerlist)
        self.available_features = set(range(num_all_features))-featureset

    def mutate(self):
        mutation_list = [
                'add_feature',
                'remove_feature',
                'add_layer',
                'remove_layer',
                'change_layer',
                'change_dropout',
                'change_reg_constant',
                'change_learning_rate',
                'change_lr_decay'
                ]
        mutation = np.random.choice(mutation_list)

        if mutation=='add_feature':
            if self.num_features<self.num_all_features:
                new_feature = np.random.choice(list(self.available_features))
                self.featureset.add(new_feature)
                self.num_features = self.num_features+1
                self.available_features = self.available_features-{new_feature}
        elif mutation=='remove_feature':
            if self.num_features>1:
                remove_feature = np.random.choice(list(self.featureset))
                self.featureset = self.featureset - {remove_feature}
                self.num_features = self.num_features-1
                self.available_features.add(remove_feature)
        elif mutation=='add_layer':
            self.layerlist.append(self.layerlist[self.num_layers-1])
            self.dropoutlist.append(0)
            self.num_layers = self.num_layers+1
        elif mutation=='remove_layer':
            if self.num_layers>1:
                self.layerlist.pop(self.num_layers-1)
                self.dropoutlist.pop(self.num_layers-1)
                self.num_layers = self.num_layers-1
        elif mutation=='change_layer':
            layer_to_change = np.random.randint(self.num_layers)
            self.layerlist[layer_to_change] = int(np.round(np.random.normal(
                    self.layerlist[layer_to_change],
                    self.std_layersize
            )))
            if self.layerlist[layer_to_change]<1:
                self.layerlist[layer_to_change]=1
        elif mutation=='change_dropout':
            dropout_to_change = np.random.randint(self.num_layers)
            self.dropoutlist[dropout_to_change] = np.random.normal(
                    self.dropoutlist[dropout_to_change],
                    self.std_dropoutrate
            )
            if self.dropoutlist[dropout_to_change]<0:
                self.dropoutlist[dropout_to_change]=0
            if self.dropoutlist[dropout_to_change]>0.5:
                self.dropoutlist[dropout_to_change]=0.5
        elif mutation=='change_reg_constant':
            self.reg_constant = self.reg_constant*np.random.choice([0.5,2])
        elif mutation=='change_learning_rate':
            self.learning_rate = self.learning_rate*np.random.choice([0.5,2])
        elif mutation=='change_lr_decay':
            self.lr_decay = self.lr_decay*np.random.choice([0.5,2])


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
