# First Attempt at Classifying 10 Animals 

I developed the following code by following along(quite closely) with the first and second lessons from fastai.

```python 
from duckduckgo_search import ddg_images
from fastbook import download_images, search_images_ddg, resize_images
from fastcore.all import*
import os

from fastbook import *

from fastai.vision.widgets import *


searches ='dog','cat','horse','lion','tiger','snake','capybara','bear', 'wolf','koala'
#
path = Path('Ten_Different_Animals')
from time import sleep

for o in searches:
    print("Searching:",o)
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images_ddg(f'{o} photo',30))
    resize_images(path/o, max_size=400, dest=path/o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
path = Path('Ten_Different_Animals')

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=5)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

this code produced the folloowing confusion matrix:

![alt test][C:/Users/Tech/Documents/GitHub/candrewdb9.github.io/images/10AnimalConfusionMatrix.png,"10 Animals Confusion Matrix"]
