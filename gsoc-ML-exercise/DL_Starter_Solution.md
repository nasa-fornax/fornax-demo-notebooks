---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: envastro
  language: python
  name: envastro
---

## Initializing the environment

```{code-cell} ipython3
import astropy.io.fits as fits
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('error')

def abmag(flux):
    return np.where(flux > 0, -2.5 * np.log10(flux) + 23.9, -99.0)
```

 ## Reading in the input files from Box

```{code-cell} ipython3
## Reading in the 1st field: GOODS-S 
url = 'https://caltech.box.com/shared/static/yad6e2mgd1rbngrsfpkod86lly1gip18'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    gs = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download GOODS-S.")
    
#these are the name of the filters we use in GOODS-S and their central wavelength
centerwave_gs = {'VIMOS_U_FLUX':3734,'ACS_F435W_FLUX':4317,'ACS_F606W_FLUX':5918,'ACS_F775W_FLUX':7617,'ACS_F814W_FLUX':8047,'ACS_F850LP_FLUX':9055,'WFC3_F098M_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'ISAAC_KS_FLUX':21600,'IRAC_CH1_FLUX':36000}
```

```{code-cell} ipython3
## Reading in the 2nd field: GOODS-N 
url = 'https://caltech.box.com/shared/static/jv3gyp0kkxnbql5wnpodujjn4cvchrud'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    gn = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download GOODS-N.")
    
centerwave_gn = {'KPNO_U_FLUX':3647,'ACS_F435W_FLUX':4317,'ACS_F606W_FLUX':5918,'ACS_F775W_FLUX':7617,'ACS_F814W_FLUX':8047,'ACS_F850LP_FLUX':9055,'WFC3_F105W_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'CFHT_KS_FLUX':21460,'IRAC_CH1_SCANDELS_FLUX':36000}
```

```{code-cell} ipython3
## Reading in the 3rd field: UDS 
url = 'https://caltech.box.com/shared/static/q8oxrb3zisw0xnekrocuydxwoivge91x'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    uds = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download UDS.")
    
centerwave_uds = {'CFHT_U_FLUX':3825,'SUBARU_B_FLUX':4500,'SUBARU_r_FLUX':5960,'ACS_F606W_FLUX':6500,'SUBARU_i_FLUX':7680,'ACS_F814W_FLUX':8047,'SUBARU_Z_FLUX':8890,'HAWKI_Y_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'HAWKI_KS_FLUX':21470,'IRAC_CH1_SEDS_FLUX':36000}
```

```{code-cell} ipython3
## Reading in the 4th field: EGS 
url = 'https://caltech.box.com/shared/static/sthjm6vl6b8bdhvg38lyursps9xnoc6h'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    egs = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download EGS.")
    
centerwave_egs = {'CFHT_U_FLUX':3825,'CFHT_G_FLUX':4810,'ACS_F606W_FLUX':5960,'CFHT_R_FLUX':6250,'CFHT_I_FLUX':7690,'ACS_F814W_FLUX':8090,'CFHT_Z_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'WIRCAM_K_FLUX':21460,'IRAC_CH1_FLUX':36000}
```

```{code-cell} ipython3
## Reading in the 5th field: COSMOS
url = 'https://caltech.box.com/shared/static/27xwurf6t3yj2i4mn0atk7gof4y22ow9'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    cos = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download COSMOS.")
    
centerwave_cos = {'CFHT_U_FLUX':3825,'SUBARU_B_FLUX':4500,'CFHT_G_FLUX':5960,'ACS_F606W_FLUX':6500,'CFHT_I_FLUX':7619,'ACS_F814W_FLUX':8047,'SUBARU_Z_FLUX':8829,'ULTRAVISTA_Y_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'ULTRAVISTA_Ks_FLUX':21521,'IRAC_Ch1_FLUX':36000}
```

 ## Plotting the SED (spectral energy distribution) of one data point
 Flux as a function of wavelength

```{code-cell} ipython3
np.random.seed(0)
randomid = np.random.randint(len(gs))
plt.figure(figsize=(6,4))
plt.title('SED of a galaxy in the GOODS-S field')
for w in centerwave_gs:
    if gs[w][randomid] > 0:  # Only plot positive flux values
        plt.plot(centerwave_gs[w],gs[w][randomid],'r*',markersize=10)

plt.yscale('log')
plt.xlabel('Wavelength(A)')
plt.ylabel('Flux (microJansky)')
plt.show()
```

##  We need output to be a catalog combined of all 5 fields with flux in the filters below:

```{code-cell} ipython3
centerwave_out = {'u':3826,'g':4877,'r':6231,'i':7618,'z':8828,'y':10214,'j':12535,'h':14453,'k':17316}
```

## Analysis
In this problem, the dataset provides flux values in certain filters. From the plot above, we can see the flux-wavelength graph is approximately on a convex curve. 

Next, let's take a look at the dataset. For example, here is the information of *GOODS-S* on given filters. There are quite a few abnormal values that we should deal with. 

```{code-cell} ipython3
import pandas as pd
pd.DataFrame(gs).describe().loc[:,centerwave_gs.keys()]
```

In this exercise, we need the flux in filters `centerwave_out`. However, the requested flux values are not explicitly given in the dataset, so we have to determine the unknown values based on the given relationship between flux and wavelength.

+++

## Simple method

Since central wavelengths of output filters are within the range of given filters, we can simply implement an interpolation method to obtain the flux values. Here, I choose cubic spline method due to its high accuracy and smoothness. Then I implement it on flux values in log scale for higher accuracy. Negative flux values are filtered out.

```{code-cell} ipython3
from scipy.interpolate import CubicSpline

# keep positive flux values
def del_neg(x, y):
    for i in reversed(range(len(x))):
        if y[i] <= 0:
            x = np.delete(x, i)
            y = np.delete(y, i)
    return x, y

def log_spline(wavelength, flux, centerwave_out):
    wavelength, flux = del_neg(wavelength, flux)
    flux_log = np.log(flux)
    flux_loginterp = CubicSpline(wavelength, flux_log) # interpolation in log scale
    flux_out = np.exp(flux_loginterp(centerwave_out))
    return flux_loginterp, flux_out
```

We can implement the method on all data points. First, I will demonstrate with one data point in each field. 

Plot notations: 
- *star* for original data points
- *cross* for required data points
- *line* for the interpolated curve

*All SED plots in this notebook are on log scale.*

```{code-cell} ipython3
fields_name = ['GOODS-S', 'GOODS-N', 'UDS', 'EGS', 'COSMOS']
fields_var = [gs, gn, uds, egs, cos]
filters_var = [centerwave_gs, centerwave_gn, centerwave_uds, centerwave_egs, centerwave_cos]
colors = ['r', 'y', 'g', 'b', 'm']
wavelengths_out = np.array(list(centerwave_out.values()))
wavelengths_interp = np.arange(3600, 36000)
plt.rcParams["legend.loc"] = "lower right"

plt.figure(figsize=(6,4))
plt.title('SED of galaxies')
for field, fieldname, filterdic, color in zip(fields_var, fields_name, filters_var, colors):
    wavelengths = np.array(list(filterdic.values()))
    fluxes = np.array([field[filtr][randomid] for filtr in filterdic])
    flux_loginterp, flux_out = log_spline(wavelengths, fluxes, wavelengths_out)
    
    plt.plot(wavelengths, fluxes, color+'*', markersize=9, label=fieldname)
    plt.plot(wavelengths_out, flux_out, color+'x', markersize=8)
    plt.plot(wavelengths_interp, np.exp(flux_loginterp(wavelengths_interp)), color)

plt.yscale('log')
plt.xlabel('Wavelength(A)')
plt.ylabel('Flux (microJansky)')
plt.legend()
plt.show()
```

For cubic spline method, the output values are close to real values though the interpolated curve does not fit very well. 

Then, I implement the method on all data points and concatenate output values into a pandas dataframe.

```{code-cell} ipython3
data_out = [[]] * 5
i = 0
for field, filterdic in zip(fields_var, filters_var):
    data_outi = np.empty(len(wavelengths_out))
    for id in range(field.shape[0]):
        try:
            wavelengths = np.array(list(filterdic.values()))
            fluxes = np.array([field[filtr][id] for filtr in filterdic])
            _, flux_out = log_spline(wavelengths, fluxes, wavelengths_out)
            data_outi = np.vstack((data_outi, flux_out))
        except ValueError as e:
            # pass galaxies with 1 or less positive flux value
            pass
        except RuntimeWarning:
            # pass overflow
            pass
    data_out[i] = data_outi
    i += 1
```

```{code-cell} ipython3
data_col1 = np.array([[field] * len(centerwave_out) for field in fields_name]).flatten()
data_col2 = np.array(list(centerwave_out.keys()) * len(fields_name))
df_list = [[]] * 5
for i in range(5):
    col_name = [fields_name[i] + '_' + key for key in list(centerwave_out.keys())]
    df_list[i] = pd.DataFrame(data=data_out[i], columns=pd.Series(col_name))
df_out = df_list[0]
for i in range(4):
    if (df_list[i].shape[0] >= df_list[i+1].shape[0]):
        df_out = df_out.join(df_list[i+1], how='left')
    else:
        df_out = df_out.join(df_list[i+1], how='right')
```

`df_out` is the output dataframe. It consists of 45 columns and 41440 rows. The column names show the field and filter names. Since the five fields do not have the same amount of data, there are some null values. 

```{code-cell} ipython3
df_out
```

Then, we can convert the dataframe to another type we want and save the file. For example, to `Table` in `astropy`.

```{code-cell} ipython3
from astropy.table import Table
file_out = Table.from_pandas(df_out)
file_out.info
```

## Machine learning - L1 regularization

Regarding to noise and errors in observations, not all data points are accurate and should be fitted perfectly. In machine learning, it's crucial to add a penalty function to prevent overfitting. Here, I implemented L1 regularization method on the dataset. 

```{code-cell} ipython3
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
X_predict = np.array(list(centerwave_out.values()))
X_all = wavelengths_interp
```

```{code-cell} ipython3
plt.figure(figsize=(6,4))
plt.title('SED of galaxies')
for field, fieldname, filterdic, color in zip(fields_var, fields_name, filters_var, colors):
    X_train = np.array(list(filterdic.values()))
    y_train = np.array([field[filtr][randomid] for filtr in filterdic])
    X_train, y_train = del_neg(X_train, y_train)
    
    # define model
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train.reshape(-1,1), y_train)
    scores = cross_val_score(lasso, X_train.reshape(-1,1), y_train, cv=5, scoring='neg_mean_squared_error')
    print(f'mean squared error for {fieldname}:\t{scores.mean()}')
    y_predict = lasso.predict(X_predict.reshape(-1,1))
    y_all = lasso.predict(X_all.reshape(-1,1))
    
    plt.plot(X_train, y_train, color+'*', markersize=9, label=fieldname)
    plt.plot(X_predict, y_predict, color+'x', markersize=8)
    plt.plot(X_all, y_all, color)

plt.yscale('log')
plt.ylim(10**(-1.8), 7)
plt.xlabel('Wavelength(A)')
plt.ylabel('Flux (microJansky)')
plt.legend()
plt.show()
```

This time, the predicted curve does not fit all data perfectly but looks like a basic function. We may find out the exact relationship by trying the combination of functions with similar shapes.

+++

## Deep Learning - multilayer perceptron

We can also solve the problem with deep learning models. Let's try a simple multilayer perceptron model with four layers, two of which are hidden. The first hidden layer has 16 nodes and the second one has 32 nodes.

```{code-cell} ipython3
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self):    
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# the following code for early stopping is from:
# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
```

To train the model, I use mean squared error as error function and stochastic gradient descent as optimizer. The largest epoch number is set to 10000 while early stopping is enabled. For simplicity, I only trained on one galaxy from GOODS-S. 

The model was trained on cpu and `manual_seed` is set for reproducibility.

```{code-cell} ipython3
torch.manual_seed(97)

field = fields_var[0]
filterdic = filters_var[0]
X_train = np.array(list(filterdic.values()))
y_train = np.array([field[filtr][randomid] for filtr in filterdic])
X_train, y_train = del_neg(X_train, y_train)

# rescale training data
X_train = X_train / 30000
y_train = y_train * 7

X_train = torch.as_tensor(X_train.reshape(-1,1)).float().to(device)
y_train = torch.as_tensor(y_train.reshape(-1,1)).float().to(device)

mlp = MLP()
mlp.to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.038, momentum=0.8)
early_stopper = EarlyStopper(patience=10, min_delta=1e-10)

# train
EPOCHS = 10000
loss_all = []
for epoch in range(EPOCHS):
    y_predict = mlp(X_train)
    loss = loss_func(y_predict, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_all.append(loss.data)
    
    if (early_stopper.early_stop(loss.data)):
        print('Epoch {0: <3}  loss={1}'.format(str(epoch+1), loss.data))
        EPOCHS = epoch + 1
        break
    if ((epoch+1) % 500 == 0):
        print('Epoch {0: <3}  loss={1}'.format(str(epoch+1), loss.data))
print('Training end.')
```

Plot training loss. 

```{code-cell} ipython3
plt.figure(figsize=(6,3))
plt.plot(range(EPOCHS), loss_all)
plt.title('Linear Regression Training Loss on GOODS-S')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()
```

Plot predicted values.

```{code-cell} ipython3
X_out = torch.as_tensor((np.array(list(centerwave_out.values())) / 30000).reshape(-1,1)).float()
y_out = mlp(X_out)
X_predict = torch.as_tensor(np.arange(0.12, 1.2, 0.01).reshape(-1,1)).float()
y_predict = mlp(X_predict)
plt.figure(figsize=(6,4))
plt.plot(X_train * 30000, y_train / 7, 'r*', markersize=10, label='real')
plt.plot(X_out.numpy() * 30000, y_out.detach().numpy() / 7, 'bx',markersize=9, label='predict (DL)')
plt.plot(X_predict.numpy() * 30000, y_predict.detach().numpy() / 7, 'b')
plt.yscale('log')
plt.rcParams['legend.loc'] = 'lower right'
plt.legend()
plt.title('SED of a galaxy in the GOODS-S field')
plt.xlabel('Wavelength(A)')
plt.ylabel('Flux (microJansky)')
plt.show()
```

The predicted curve goes between real data points, which is what we expected.
