# Methods for Exploratory Data Analysis
Exploratory data analysis is an approach to analysing the data to summarise there main characteristics. This can help evaluate datasets, and I'll be using EDA to critque potential data augmentations. In this post I will be Analysing the techniques based on the *UCI ML hand-written digits datasets*. The following three methods seem to be popular approaches, and will be the ones analyseed here.

## Method 1: Multidimensional scaling (MDS)
MDS is also known as Principal Coorinate Analysis(PCoA), but since I'll be discussing PCA later I will Stick to MDS.MDS provides a visual representation of dissimilarities between classes in the dataset. The MDS Algorithm works to minimize the loss function, called *strain*, for an input matrix.

$$
Strain(x_1,x_2,...x_n) = ((\sum_{i,j} (b_ij -x^T_i x_j)^2)/(\sum_{i,j} b_ij ^2) ) 
$$

## Method 2: PCA


## Method 3: T-SNE