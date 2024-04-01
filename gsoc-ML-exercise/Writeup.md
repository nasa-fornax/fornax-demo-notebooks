# GSoC2024 DL toy problem writeup

## Current work

My work can roughly be divided into two parts. One is learning about `astropy` and the provided dataset, and the other is implementing diffrent methods to meet the requirements.

I finished the required and the first optional assignments. To combine all five fields in requested wavelengths, I first implemented interpolation and regression algorithms to obtain flux values. The three methods I worked on are cubic spline, L1 regularization, and linear regression. All methods work well while L1 regularization and linear regression outperforms cubic spline.

After getting the required data, I combined them in one pandas dataframe which can easily be converted to output files or astropy data types.

## Future work

For ML and DL approaches, I only implemented on part of the dataset. Training on the entire data will be completed if I have more time. Also, I will look into the physics fundations behind the data to determine which prediction is more accurate. In the deep learning model, the output prediction can change from being linear to fitting the original data perfectly. Although the current prediction looks nice, I still need more time to optimize the model and may try other regression methods apart from linear regression. 

What's more, I will spend more time in the dataset. The reason I didn't finish the second optional requirement is that I'm not sure about the way redshifts or stellarmasses are represented. Then I may use some clustering methods to group the galaxies. 

