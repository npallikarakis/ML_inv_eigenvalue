print('----------- MLP for the inverse Transmission Eigenvalue problem for picewise constant refractive index -------------------')
print('----------- N. Pallikarakis and A. Ntargraras - https://arxiv.org/abs/2212.04279 ----------------')
print('----------- Copyright (C) 2023 N. Pallikarakis ----------------------------------')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow import random
from tensorflow.keras import regularizers


#############function for perturbation rank
def perturbation_rank(model, x, y, names, regression):
    errors = []
    from sklearn import metrics
    import pandas as pd

    for i in range(x.shape[1]):
        hold = np.array(x[:, i])
        np.random.shuffle(x[:, i])

        if regression:
            pred = model.predict(x)
            error = metrics.mean_squared_error(y, pred)
        else:
            pred = model.predict_proba(x)
            error = metrics.log_loss(y, pred)

        errors.append(error)
        x[:, i] = hold

    max_error = np.max(errors)
    importance = [e / max_error for e in errors]

    data = {'name': names, 'error': errors, 'importance': importance}
    result = pd.DataFrame(data, columns=['name', 'error', 'import'
                                                          'ance'])
    result.sort_values(by=['importance'], ascending=[0], inplace=True)
    result.reset_index(inplace=True, drop=True)
    return result

###########function for regression chart
def chart_regression(pred, y, sort=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), marker='o', label='expected')
    plt.plot(t['pred'].tolist(), marker='o', label='prediction')
    plt.ylabel('output MLP')
    plt.legend()
    plt.show()


random.set_seed(42)

model = Sequential([
    Dense(800, activation = 'relu',activity_regularizer=regularizers.l2(2e-4)),
    #Dropout(0.5, seed=42),
    #Dense(200, activation = 'relu',activity_regularizer=regularizers.l2(2e-4)),
    Dense(200, activation = 'relu',activity_regularizer=regularizers.l2(2e-4)),
    #Dropout(0.3, seed=42),
    Dense(100, activation = 'relu',activity_regularizer=regularizers.l2(2e-4)),
    #extra layer
    Dense(50, activation = 'relu',activity_regularizer=regularizers.l2(2e-4)),
    # Dropout(0.5, seed=42),
    Dense(3, activation = None)
])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=RootMeanSquaredError())  # adam or SGD

# early stopping
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=100, verbose=1, mode='auto', restore_best_weights=True)


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy as np
from sklearn import metrics

######### load data
train_data = np.loadtxt('classic_train.csv', delimiter=',',skiprows=1)
#original data coming from separation of variables in discs
real_data = np.loadtxt('classic_original_10.csv', delimiter=',',skiprows=1) ##ok

print(train_data.shape)
print(real_data.shape)

########## random shuffle of the train data
np.random.seed(100)
#np.random.seed(5)
#np.random.seed(105)
#np.random.seed(205)
#np.random.seed(305)
#np.random.seed(405)
#np.random.seed(505)
#np.random.seed(605)
#np.random.seed(705)
#np.random.seed(805)
#np.random.seed(10) #in place of seed 705
np.random.shuffle(train_data)


######## Split them into dependent and independent variables
### training data#####
# the lowest 6 real eigenvalues
X_tr = train_data[:,1:7] #classic & mod
# the refractive index n1,n2 and discontinuiity d1
Y_tr = train_data[:,10:13] #classic


X_test = real_data[:,0:6]
Y_test = real_data[:,6:9]

print(Y_tr.shape)
print(X_tr.shape)
print(X_test.shape)
print(X_test.shape)


##### Prerocessing Step

from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split

# Splitting train data into train and valid(test) set to scale them avoiding leaking of data
# The valid(test) set is held back in order to provide unbiased evaluation during a models hyperparameter tuning
X_train, X_val, y_train, y_val = train_test_split(X_tr, Y_tr, test_size=0.3, random_state=10)

#### Standardization, or mean removal and variance scaling
sscaler_X = preprocessing.StandardScaler()
X_train_ss = sscaler_X.fit_transform(X_train)
X_val_ss = sscaler_X.transform(X_val)
X_test_ss = sscaler_X.transform(X_test)


#################################### MLP #######################
X = X_train_ss
y= y_train


history = model.fit(X_train_ss, y_train, validation_data=(X_val_ss, y_val), callbacks=[monitor], verbose=2,epochs=8000)


Y_predict = model.predict(X_test_ss)
Y_predict_tr = model.predict(X_train_ss)
Y_predict_val = model.predict(X_val_ss)

print('y_pred',Y_predict)
print('y_original',Y_test)

#np.savetxt('results_2L_MLP_ss.csv', Y_predict,delimiter=',')

########print predicted y vs real y
chart_regression(Y_predict[:,0].flatten(), Y_test[:,0])
chart_regression(Y_predict[:,1].flatten(), Y_test[:,1])
chart_regression(Y_predict[:,2].flatten(), Y_test[:,2])


print('----------- r2 scores -------------------')
from sklearn.metrics import r2_score
print('R squared training set r2', round(metrics.r2_score(y_train, Y_predict_tr)*100, 2))
print('R squared validating set r2', round(metrics.r2_score(y_val, Y_predict_val)*100, 2))
print('R squared testing set r2', round(metrics.r2_score(Y_test, Y_predict)*100, 2))
from sklearn.metrics import r2_score


print('----------- r2 score individually -------------------')
#print('R squared real y set', round(r2_score(Y_re,Y_predict, )*100,2))
print('R squared real n1 set', round(r2_score(Y_test[:,0],Y_predict[:,0])*100,2))
print('R squared real n2 set', round(r2_score(Y_test[:,1],Y_predict[:,1])*100,2))
print('R squared real r1 set', round(r2_score( Y_test[:,2],Y_predict[:,2])*100,2))


###########ranking of eigenvalues
#"""
print('----------- rankings -------------------')
columns = X_tr.shape[1]
names = [i for i in range(1,columns+1)]

#names = list(['k1', 'k2', 'k3', 'k4', 'k5', 'k6'])
#train set
rank = perturbation_rank(model, X, y, names, True)
print(rank)

#np.savetxt('rank_MLP_2L_ss.csv', rank,delimiter=',')
#"""

print(model.summary())