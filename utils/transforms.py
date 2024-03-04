import torch
from torch_geometric.transforms import BaseTransform, RandomRotate
from torch_geometric.data import Data


class SequentialTransformation(BaseTransform):
    """
    Takes the sequence of transformations and applies it to a data point.
    """

    def __init__(self, transforms: list):
        assert isinstance(transforms, list)
        self.transforms = transforms

    def __call__(self, data: Data) -> Data:
        for transform in self.transforms:
            data = transform(data)
        return data

    def __str__(self):
        return "{ " + "\n".join([str(t) for t in self.transforms]) + " }"

    def __getitem__(self, index):
        return self.transforms[index]


class NodeFeatSubset(BaseTransform):
    """
    Selects a subset of node feature variables from an object.
    """

    def __init__(self, indices: list):
        assert isinstance(indices, list)
        self.indices = indices

    def __call__(self, data: Data) -> Data:
        data.x = data.x[:, self.indices].view(-1, len(self.indices))
        return data

    def __str__(self):
        return (
            "subset of node features: { "
            + "\n".join([str(ind) for ind in self.indices])
            + " }"
        )


class TargetSubset(BaseTransform):
    """
    Selects a subset of target variables from an object.
    """

    def __init__(self, indices: list):
        assert isinstance(indices, list)
        self.indices = indices

    def __call__(self, data: Data) -> Data:
        data.y = data.y[:, self.indices].view(-1, len(self.indices))
        return data

    def __str__(self):
        return (
            "subset of targets: { "
            + "\n".join([str(ind) for ind in self.indices])
            + " }"
        )


class TargetNormalize(BaseTransform):
    """
    Normalizes target variables by substrating its mean and dividing by its standard deviation.
    """

    def __init__(self, mean: list, std: list):
        self.mean = torch.Tensor(mean).view(1, 1)
        self.std = torch.Tensor(std).view(1, 1)

    def __call__(self, data: Data) -> Data:
        assert data.y.shape[-1] == self.mean.shape[-1] == self.std.shape[-1]
        data.y = (data.y - self.mean) / self.std
        return data


class RemoveEdges(BaseTransform):
    """
    Normalizes target variables by substrating its mean and dividing by its standard deviation.
    """

    def __call__(self, data: Data) -> Data:
        data.edge_index = None
        data.edge_attr = None
        return data


class RandomRotateAxes(BaseTransform):
    """
    Rotates node positions around specific axes by 360°.
    """

    def __init__(self, axes: tuple):
        self.axes = axes
        if axes is None:
            self.rotators = [lambda x: x]
        else:
            self.rotators = []
            for axis in axes:
                assert axis in [0, 1, 2]
                self.rotators.append(RandomRotate((-180, 180), axis))

    def __call__(self, data: Data) -> Data:
        for rotator in self.rotators:
            data = rotator(data)
        return data

    def __str__(self):
        return "RandomRotate around axes: " + ", ".join(
            [str(axis) for axis in self.axes]
        )
