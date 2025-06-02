"""
This module contains an abstract class for classification datasets, and
provides a uniform way to load datasets from different sources for local
experiments.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Type

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class BaseDataset(ABC):
    """
    Abstract class for classification datasets.

    Provides a uniform way to load datasets from different sources for local
    experiments. The class is designed to be subclassed, and the `load` method
    must be implemented by subclasses to provide the actual dataset loading
    functionality.

    Attributes:
        name (str): The name of the dataset.
        X (iterable): The input data.
        y (iterable): The target labels.
        X_train (iterable): The training input data.
        y_train (iterable): The training target labels.
        X_val (iterable): The validation input data.
        y_val (iterable): The validation target labels.
        X_test (iterable): The testing input data.
        y_test (iterable): The testing target labels.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __init__(
        self,
        name: str,
        train_size: float = 0.7,
        validation_size: float = 0.1,
        one_hot: bool = True,
    ):
        """
        Initialize the dataset with a name.

        Args:
            name (str): The name of the dataset.
            train_size (float): The proportion of the dataset to include in the
                training set. Default is 0.7.
            validation_size (float): The proportion of the dataset to include in
                the validation set. Default is 0.1.
            one_hot (bool): If True, the target labels will be one-hot encoded.
                Default is True.
        """
        self.name = name

        self.load()
        if one_hot:
            self.y = one_hot_encode(self.y)
        self.split(train_size, validation_size)

    @property
    def n_classes(self):
        """
        Get the number of classes in the dataset.

        Returns:
            int: The number of classes in the dataset.
        """
        return self.y.shape[1] if len(self.y.shape) > 1 else len(np.unique(self.y))

    @abstractmethod
    def load(self):
        """
        Load the dataset.

        This method sets the `X` and `y` attributes of the dataset instance
        with the input data and target labels, respectively.

        Returns:
            tuple: A tuple containing the training and testing data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def split(self, train_size: float, validation_size: float):
        """
        Split the dataset into training, validation, and testing sets.

        Args:
            train_size (float): The proportion of the dataset to include in the
                training set.
            validation_size (float): The proportion of the dataset to include in
                the validation set.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, train_size=train_size + validation_size, stratify=self.y
        )

        # Calculate the adjusted validation size
        adjusted_validation_size = validation_size / (train_size + validation_size)

        # Split the training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, train_size=1 - adjusted_validation_size, stratify=y_train
        )

        # Set the training, validation, and testing sets
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test


# -----------------------------------------------------------
# Utility functions for loading datasets
# -----------------------------------------------------------
def one_hot_encode(y):
    """
    One-hot encode the target labels.

    Args:
        y (iterable): The target labels.

    Returns:
        np.ndarray: The one-hot encoded labels.
    """
    encoder = OneHotEncoder(sparse_output=False)
    y = np.array(y).reshape(-1, 1)
    return encoder.fit_transform(y)


# -----------------------------------------------------------
# Implementations and Factory
# -----------------------------------------------------------
class DatasetFactory:
    """
    Simple string-to-class mapping for dataset loading.
    """

    _registry: Dict[str, Type[BaseDataset]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseDataset]], Type[BaseDataset]]:
        """
        Decorator to register a dataset class with a name.

        Args:
            name (str): The name of the dataset.
        """

        def decorator(dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
            cls._registry[name] = dataset_class
            return dataset_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDataset:
        """
        Create an instance of a registered dataset class.

        Args:
            name (str): The name of the dataset.
            **kwargs: Additional keyword arguments to be passed to the
                dataset class constructor
        """
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered.")
        dataset_class = cls._registry[name]
        return dataset_class(**kwargs)


@DatasetFactory.register("moons")
class MoonsDataset(BaseDataset):
    def __init__(
        self, n: int = 1000, noise: float = 0.1, random_state: int = 42, **kwargs
    ):
        self.n = n
        self.noise = noise
        self.random_state = random_state
        super().__init__(name="moons", **kwargs)

    def load(self):
        from sklearn.datasets import make_moons

        self.X, self.y = make_moons(
            n_samples=self.n, noise=self.noise, random_state=self.random_state
        )


@DatasetFactory.register("blobs")
class BlobsDataset(BaseDataset):
    def __init__(
        self,
        n: int = 1000,
        dim: int = 2,
        num_blobs: int = 6,
        random_state: int = 42,
        **kwargs,
    ):
        self.n = n
        self.dim = dim
        self.num_blobs = num_blobs
        self.random_state = random_state
        super().__init__(name="blobs", **kwargs)

    def load(self):
        from sklearn.datasets import make_blobs

        self.X, self.y = make_blobs(
            n_samples=self.n,
            n_features=self.dim,
            centers=self.num_blobs,
            random_state=self.random_state,
        )


@DatasetFactory.register("circles")
class CirclesDataset(BaseDataset):
    def __init__(
        self, n: int = 1000, noise: float = 0.1, random_state: int = 42, **kwargs
    ):
        self.n = n
        self.noise = noise
        self.random_state = random_state
        super().__init__(name="circles", **kwargs)

    def load(self):
        from sklearn.datasets import make_circles

        self.X, self.y = make_circles(
            n_samples=self.n, noise=self.noise, random_state=self.random_state
        )


@DatasetFactory.register("iris")
class IrisDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(name="iris", **kwargs)

    def load(self):
        from sklearn.datasets import load_iris

        self.X, self.y = load_iris(return_X_y=True)


@DatasetFactory.register("mnist")
class MNISTDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(name="mnist", **kwargs)

    def load(self):
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1)
        self.X = mnist.data.to_numpy().astype(np.float32)
        self.X = self.X / 255.0
        self.y = mnist.target.to_numpy()


@DatasetFactory.register("rice")
class RiceDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(name="rice", **kwargs)

    def load(self):
        from ucimlrepo import fetch_ucirepo

        rice = fetch_ucirepo(id=545)
        self.X = rice.data.features.to_numpy()
        self.y = rice.data.targets.to_numpy()


if __name__ == "__main__":
    ds = MNISTDataset()
    print(ds.X.shape, ds.X.dtype)
    print(ds.y.shape, ds.y.dtype)
    print(f"Train: {ds.X_train.shape}, {ds.y_train.shape}")
    print(f"Validation: {ds.X_val.shape}, {ds.y_val.shape}")
    print(f"Test: {ds.X_test.shape}, {ds.y_test.shape}")
    print(ds.X[:5])
    print(ds.y[:5])
