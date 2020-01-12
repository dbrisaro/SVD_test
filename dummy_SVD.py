"""
Test of SVD with dummy variables

@dbrisaro
Jan 2020
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# lets consider two fields with time in the first dimension
# and latitude/longitude in the second and third dimension
# (I'll use random arrays)
S = np.random.rand(72,50,50)
T = np.random.rand(72,50,50)

# size of each dimension
ntime, nlat, nlon = S.shape

# reshape S and T arrays into a 2D matrix of dimensions ntime x (nlat*nlon)
S_res = np.reshape(S, (ntime, nlat*nlon))
T_res = np.reshape(T, (ntime, nlat*nlon))

# remove the time mean of each field, i.e compute anomalies
S_anom = (S_res - np.mean(S_res, axis=0))
T_anom = (T_res - np.mean(T_res, axis=0))

# we'll perform the SVD analysis to the covariance matrix C
C = np.dot(np.transpose(S_anom), T_anom)
C = C/(ntime-1)         # I'm not sure if this step is absolutely necessary

# SVD
u, l, v = np.linalg.svd(C)

# now we will compute regular SVD calculations to display them in a figure

# a) Explained variance: this info is stored in the diagonal matrix l
# which contains the eigenvalues of each mode. The method makes it
# easier and returns directly a vector (1D array).
scf = np.power(l, 2)/np.sum(np.power(l, 2))*100

# b) Pairs of coupled spatial patterns, we'll take the first mode
# ie, first pair of spatial patterns
# in U, the spatial patterns are stored in the columns while in V in rows
# I'll make a reshape to recover the shape nlat*nlon
svd_S1 = np.reshape(u[:,0], (nlat,nlon))
svd_T1 = np.reshape(v[0,:], (nlat,nlon))

# c) The temporal variation it's represented by the expansion coefficients (PC)
# that are the projection of the original data into the singular matrixes (U,V)
A = np.dot(S_anom, u)                    # PCs are in columns
B = np.dot(v, np.transpose(T_anom))      # PCs are in rows

# for the first mode
PC_S1 = (A[:,0] - np.mean(A[:,0])) / np.std(A[:,0])     #standardized
PC_T1 = (B[0,:] - np.mean(B[0,:])) / np.std(B[0,:])

# d) Sometimes is not easy to interpret the amplitudes of the SVD spatial fields.
# In this case we'll provide this information through correlation maps.
# We'll compute the so-called 'homogeneous maps' defined as the vector
# correlation between the expansion coefficient of the k-th mode of a field
# and the same values of the same field at each grid point.
# In our case, the k-th mode will be the first mode. If we take the S field,
# the homogeneous map will be the correlation between svd_S1 and PC_S1
# (and also the correlation between svd_T1 and PC_T1).

corr_S1 = np.zeros((nlat*nlon))
corr_T1 = np.zeros((nlat*nlon))
for i in range(nlat*nlon):
    corr_S1[i] = np.corrcoef(S_anom[:,i], PC_S1)[0,1]
    corr_T1[i] = np.corrcoef(T_anom[:,i], PC_T1)[0,1]
    # I'm more than sure there is a better way to calculate this, but im pretty dull

# final reshape
SVD_S1_hom = np.reshape(corr_S1, (nlat,nlon))
SVD_T1_hom = np.reshape(corr_T1, (nlat,nlon))

#-------------------------------------------------------
# The final plot

plt.style.use('ggplot')     # I like the ggplot style!

title = ['a) Correlation map S1',
        'b) Correlation map T1',
        'c) Expansion coefficients (PCs)',
        'd) Squared covariance \n fraction (SCF)']

position = [[0.05, 0.35, 0.42, 0.60],
            [0.55, 0.35, 0.42, 0.60],
            [0.05, 0.05, 0.64, 0.24],
            [0.75, 0.05, 0.22, 0.24]]

colorbar = [0.985, 0.45, 0.01, 0.40]

cmap = plt.cm.RdBu_r

fontsize = 5
plt.rcParams.update({'font.size': fontsize})

figsize = (4.5, 4)
figname = '/home/daniu/Imágenes/SVD_for_beginners.png'

fig = plt.figure(figsize=figsize)
ax = plt.axes(position[0])
im1 = ax.imshow(SVD_S1_hom, vmin=-1, vmax=1, cmap=cmap)
ax.set_title(title[0], loc='left', fontsize=5)
ax.grid(lw=0)

bx = plt.axes(position[1])
im2 = bx.imshow(SVD_T1_hom, vmin=-1, vmax=1, cmap=cmap)
bx.set_title(title[1], loc='left', fontsize=5)
bx.grid(lw=0)

cbax = fig.add_axes(colorbar)
cb = fig.colorbar(im2, orientation='vertical', cax=cbax)

cx = plt.axes(position[2])
cx.plot(PC_S1, lw=.5, color='green', label='PC S1')
cx.plot(PC_T1, lw=.5, color='purple', label='PC T1')
cx.legend()
cx.set_title(title[2], loc='left', fontsize=5)
cx.grid(lw=0.5)

dx = plt.axes(position[3])
dx.plot(scf[0:10], '-o', lw=.5, color='salmon', markersize=3)
dx.set_title(title[3], loc='left', fontsize=5)
dx.grid(lw=0.5)

fig.savefig(figname, dpi=300, bbox_inches='tight')
