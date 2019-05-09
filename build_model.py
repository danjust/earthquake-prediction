import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

class model():
    """Model class"""

    def build(self, num_features, layerlist, dropoutlist):
        """Function to create the model architecture"""
        num_layers = len(layerlist)
        model = Sequential()
        if num_layers==0:
            model.add(Dense(
                    units=1,
                    input_dim=num_features,
                    activation='relu',
                    use_bias=False,
            ))
        else:
            model.add(Dense(
                    units=layerlist[0],
                    input_dim=num_features,
                    activation='relu',
            ))
            if dropoutlist[0]>0:
                model.add(Dropout(rate=dropoutlist[0]))

            for i in range(1,num_layers):
                model.add(Dense(
                        units=layerlist[i],
                        activation='relu',
                ))
                if dropoutlist[i]>0:
                    model.add(Dropout(rate=dropoutlist[i]))
                    
            model.add(Dense(
                    units=1,
                    activation='relu',
                    use_bias=False,
            ))
        return model
