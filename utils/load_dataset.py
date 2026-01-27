import kagglehub
import os

path = kagglehub.dataset_download("yasserh/twitter-tweets-sentiment-dataset",force_download=True)
print(path)
print(os.listdir(path))