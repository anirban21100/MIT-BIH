# MIT-BIH
MIT-BIH Arrhythmia Classification

This code uses CNN to classify 5 classes of arrhythmia from MIT-BIH arrhythmia database.

The CSV Dataset used in this code can be found in: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

1. MIT_BIH2img.py : converts csv file into image file of 178x178 and save dataset.
2. Preprocessing.py : loads the previously saved dataset and does preprocessing
3. Final Model.py : the final model for MIT-BIH, accuracy = 98%
4. evaluate performance.py : evaluates the performance of the final model
