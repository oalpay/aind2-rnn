import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    for window_end in range(window_size,len(series)):
        X.append(series[window_end-window_size:window_end])
        y.append(series[window_end])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    import keras

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))


    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()

### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # remove as many non-english characters and character sequences as you can 
    import string
    remove_punctuation_map = dict((ord(char), ' ') for char in (string.punctuation + '\n'))
    text = text.translate(remove_punctuation_map)

    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    for window_end in range(window_size,len(text),step_size):
        inputs.append(text[window_end-window_size:window_end])
        outputs.append(text[window_end])     
    
    return inputs,outputs
