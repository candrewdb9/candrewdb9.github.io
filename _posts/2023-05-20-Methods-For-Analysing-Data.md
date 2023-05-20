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

## Method 2: PCA
PCA is a form of dimensionality reducuction for large data sets. "Its idea is simple—reduce the dimensionality of a dataset, while preserving as much ‘variability’ (i.e. statistical information) as possible"[[1]](#1). PCA works on analysing *n* entities, with each entity having P *numerical* observations or features. To start we need to define *p* *n*-dimensional vectors, forming a matrix *X* which defines our data. Each coloumn of *X* is the *n*-dimensional vector. Practically for image analysis this means flattening each *m*x*n* image matrix to a 1x$(m*n)$ vector, then each image vector is a row of the *X* matrix and each cloumn is a feature, or pixel, of each image in the dataset(see image below for a clearer explanation).

![alt text](https://github.com/candrewdb9/candrewdb9.github.io/raw/master/images/PCA_diagram1.png "Step 1")

![alt text](https://github.com/candrewdb9/candrewdb9.github.io/raw/master/images/PCA_diagram2.png "Step 2")

![alt text](https://github.com/candrewdb9/candrewdb9.github.io/raw/master/images/PCA_diagram3.png "Step 3")


## Method 3: T-SNE

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