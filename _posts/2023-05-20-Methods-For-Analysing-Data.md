# Methods for Exploratory Data Analysis
Exploratory data analysis is an approach to analysing the data to summarise there main characteristics. This can help evaluate datasets, and I'll be using EDA to critque potential data augmentations. In this post I will be Analysing the techniques based on the *UCI ML hand-written digits datasets*. The following three methods seem to be popular approaches, and will be the ones analyseed here.

p.s. See imports and loading the data set at the end of the post.

## Method 1: Multidimensional scaling (MDS)
MDS is also known as Principal Coorinate Analysis(PCoA), but since I'll be discussing PCA later I will Stick to MDS.MDS provides a visual representation of dissimilarities between classes in the dataset. The MDS Algorithm works to minimize the loss function, called *strain*, for an input matrix.

$$
Strain(x_1,x_2,...x_n) = (\dfrac{\sum_{i,j} (b_ij -x^T_i x_j)^2}{\sum_{i,j} b_ij ^2} ) 
$$

I tested MDS on the Digits dataset in python:

```python
# init MDS object 
embedding = MDS(n_components=2)

# Apply MDS fit to the data
MDS_embedding = embedding.fit_transform(X,y)

# Display the results
df = pd.DataFrame()
df['MDSx'] = MDS_embedding[:,0]
df['MDSy'] = MDS_embedding[:,1]

sns.scatterplot(
    x='MDSx', y='MDSy',
    hue=y,
    palette=sns.color_palette('hls',10),
    data=df,
    legend="full",
    alpha=1,
    
)
```

The above code rendered the following results.

![alt text](https://github.com/candrewdb9/candrewdb9.github.io/raw/master/images/MDS.png "MDS")

It can be seen that MDS does a good job of seperating the classes in the data, displaying information like:

* the digit *3* is close in euclidean space to *9* and *2*
* the digit *1* is close in euclidean space to *7* and *4*
* the digit *5* and *8* seem to have higher variance in euclidean space 

## Method 2: PCA
PCA is a form of dimensionality reducuction for large data sets. "Its idea is simple—reduce the dimensionality of a dataset, while preserving as much ‘variability’ (i.e. statistical information) as possible"[[1]](#1). PCA works on analysing *n* entities, with each entity having P *numerical* observations or features. To start we need to define *p* *n*-dimensional vectors, forming a matrix *X* which defines our data. Each coloumn of *X* is the *n*-dimensional vector. Practically for image analysis this means flattening each *m*x*n* image matrix to a 1x$(m*n)$ vector, then each image vector is a row of the *X* matrix and each cloumn is a feature, or pixel, of each image in the dataset(see image below for a clearer explanation).

![alt text](https://github.com/candrewdb9/candrewdb9.github.io/raw/master/images/PCA_diagram1.png "Step 1")

![alt text](https://github.com/candrewdb9/candrewdb9.github.io/raw/master/images/PCA_diagram2.png "Step 2")

![alt text](https://github.com/candrewdb9/candrewdb9.github.io/raw/master/images/PCA_diagram3.png "Step 3")

After this we define $x_i$ as the i-th coloumn for matrix *X*, and S the covariance matrix of the dataset where:

$$
S = \dfrac{1}{n - 1}\sum_{i=1}^n (x_i-\bar{X})(x_j-\bar{X})
$$

We use the covariance matrix as we seek the coloumns of *X* with the highest variance, this leads to the equation where a is a *p*-dimensional vector of constants:

$$
Sa - \lambda a =0
$$
or
$$
Sa=\lambda a
$$
from this equation we see that a is the eigenvector and $\lambda$ is the eigenvalues of the matrix *S*. The vector is the PC loadings of the dataset and the equation $Xa_k% gives us the PCA of the Data set.

Below is an example in python.
```python
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X,y)
df['PCAx'] = pca_result[:,0]
df['PCAy'] = pca_result[:,1]
sns.scatterplot(
    x='PCAx', y='PCAy',
    hue=y,
    palette=sns.color_palette('hls',10),
    data=df,
    legend="full",
    alpha=1,
    
)
```
![alt text](https://github.com/candrewdb9/candrewdb9.github.io/raw/master/images/PCA.png "PCA")

PCA gives similar results to MDS which is expected as the variance is highest in euclidean space which is what MDS measures, but it seems to group some classes tighter than others which may suggest other variable affecting the classes.


## Method 3: T-SNE
T-SNE stands for T-distributed stochastic neighbor embedding. unlike PCA and MDS T-SNE is a non-linear dimensional reduction of the dataset where sililar objects are located close together. T-SNE works to find a 2-dimensional map $Y = [y_1,y_2,...y_n]$ where $y_i \in \mathbb{R}^2$ to do so we need to define $p_{ij}$ and $q_{ij} 
$$
p_{j|i} = \dfrac{exp(-\\|x_i - x_j\\|^2 / 2\sigma_{i}^2)}{\sum_{k\neq i}exp(-\\|x_i - x_j\\|^2 / 2\sigma_{i}^2)}
$$
$$
p_{ij} = \dfrac{p_{j|i}+p_{i|j}}{2N}
$$

$p_{ij}$ are the probabilities which are proportional to the similarity between two objects $x_i$ and $x_j$.

$$
q_{ij} = \dfrac{(1+ \\|y_i - y_j\\|^2)^-1}{\sum_{k} \sum_{l\neq k} (1+ \\|y_i - y_j\\|^2)^-1}
$$

$q_{ij}$ measures the similarities betweens points on the map $Y$. T-SNE uses gradient deescent to minimize the following equation by varying the values of the map $Y$.
$$
KL(P\\|Q) = \sum_{i\neq j} p_{ij}\log{\dfrac{p_{ij}}{q_{ij}}}
$$
It is suggested that T-SNE be run on the PCA data to reduce the dimensionality of the search space which will speed up the results.
```python
tsne = TSNE(n_components=2, perplexity=100, n_iter=5000)
tsne_pca_results = tsne.fit_transform(pca_result)
df['TSNEx'] = tsne_pca_results[:,0]
df['TSNEy'] = tsne_pca_results[:,1]
sns.scatterplot(
    x='TSNEx', y='TSNEy',
    hue=y,
    palette=sns.color_palette('hls',10),
    data=df,
    legend="full",
    alpha=1,
    
)
```
![alt text](https://github.com/candrewdb9/candrewdb9.github.io/raw/master/images/TSNE.png "TSNE")

T-SNE gives far more information on the data set. it can be seen that although 0 and 6 are the close together they are easy to distiguish, 4 and 7 are close together and are also easy to distingush. But 5 and 8 are close together and hard to distigush. there are also some elements in the data set which don't sit with the rest of their cohort, this could mean they need to removed from the set, or more augmentation needs to be applied to succesfully identify them. 


## Summary
These metric can be thought to be proportional to the success of AIs interpreting the data. If the classes are hard to distiguish in these forms you may need more features to describe them. T-SNE seems to be the best metric and since PCA is a part of it it may give more data to veiw, maybe implying the level of *deepness* of the model. If PCA is able to find a linear relation a simpler model should be sufficient.  

# Loading Dataset and libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import load_digits


#MDS
from sklearn.datasets import load_digits
from sklearn.manifold import MDS

# PCA
from sklearn.decomposition import PCA

#TSNE
from sklearn.manifold import TSNE

X,y = sklearn.datasets.load_digits(return_X_y=True)

```


## References
<a id="1">[1]</a>
Jolliffe IT, Cadima J. Principal component analysis: a review and recent developments. Philos Trans A Math Phys Eng Sci. 2016 Apr 13;374(2065):20150202. doi: 10.1098/rsta.2015.0202. PMID: 26953178; PMCID: PMC4792409.