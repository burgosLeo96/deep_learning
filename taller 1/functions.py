import numpy as np

def segment(dataset, variable, window = 5000, future = 0):
        data = []
        labels = []
        for i in range(len(dataset)):
            start_index = i
            end_index = i + window
            future_index = i + window + future
            if future_index >= len(dataset):
                break
            data.append(dataset[variable][i:end_index])
            labels.append(dataset[variable][end_index:future_index])
        return np.array(data), np.array(labels)
def normalize(x, stats):
        return (x - stats['mean']) / stats['std']