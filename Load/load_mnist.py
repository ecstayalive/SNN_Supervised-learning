import os.path
import pickle
import os
import numpy as np
import gzip

class LoadMnist:
    def __init__(self):
        self.key_file = {
            'train_img': 'train-images-idx3-ubyte.gz',
            'train_label': 'train-labels-idx1-ubyte.gz',
            'test_img': 't10k-images-idx3-ubyte.gz',
            'test_label': 't10k-labels-idx1-ubyte.gz'
        }
        self.dataset_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_file = self.dataset_dir + "/mnist.pkl"
        self.train_num = 60000
        self.test_num = 10000
        self.img_dim = (1, 28, 28)
        self.img_size = 784

    def load_mnist(self, normalize=True, flatten=True, one_hot_label=False):
        """Read into the MNIST dataset

            Parameters
            ----------
            normalize : Normalize the pixel values of the image to 0.0 ~ 1.0
            one_hot_label :
                When one_hot_label is True, the label is returned as one-hot array
                 One-hot array is an array like [0,0,1,0,0,0,0,0,0,0]
            flatten : Whether to expand the image into a one-dimensional array

            Returns
            -------
           (Training image, training label), (test image, test label)
            """
        if not os.path.exists(self.save_file):
            self.init_mnist()

        with open(self.save_file, 'rb') as f:
            dataset = pickle.load(f)

        if normalize:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0

        if one_hot_label:
            dataset['train_label'] = self.change_one_hot_label(dataset['train_label'])
            dataset['test_label'] = self.change_one_hot_label(dataset['test_label'])

        if not flatten:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

        return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

    def init_mnist(self):
        dataset = self.convert_numpy()
        print("Creating pickle file ...")
        with open(self.save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)
        print("Done!")

    def convert_numpy(self):
        dataset = {}
        dataset['train_img'] = self.load_img(self.key_file['train_img'])
        dataset['train_label'] = self.load_label(self.key_file['train_label'])
        dataset['test_img'] = self.load_img(self.key_file['test_img'])
        dataset['test_label'] = self.load_label(self.key_file['test_label'])

        return dataset

    def change_one_hot_label(self, X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1

        return T

    def load_label(self, file_name):
        file_path = self.dataset_dir + "/" + file_name

        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print("Done")

        return labels

    def load_img(self, file_name):
        file_path = self.dataset_dir + "/" + file_name

        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, self.img_size)
        print("Done")

        return data
