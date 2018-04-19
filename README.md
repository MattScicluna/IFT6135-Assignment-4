# IFT 6135 Assignment 4

Directory structure
-------------------
    ├── data
    |   └── celebA
    |   └── resized_celebA
    ├── utils
    |   └── process_imgs.py
    ├── models
    |   └── model.py
    |   └── model_nn.py
    |   └── model_bilinear.py
    |   ├── transpose
    |   ├── nn
    |   ├── bilinear
    ├── results
    |   ├── generated_images
	|   |   ├── transpose
	|   |   ├── nn
	|   |   ├── bilinear
    |   ├── training_summaries
    |   |   ├── transpose
	|   |   ├── nn
	|   |   ├── bilinear



    


Download the data by running
```python download_dataset.py```

Process the images by running
```process_imgs.py```

You can remove the `img_align_celeba/` folder once this is done to save yourself 1.4GB of space.
