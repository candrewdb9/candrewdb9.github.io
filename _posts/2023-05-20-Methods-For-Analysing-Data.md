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


## Method 3: T-SNE

# Loading Dataset and libraries