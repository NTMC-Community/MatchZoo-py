import matchzoo as mz
from matchzoo.dataloader import Dataset


class DatasetBuilder(object):
    """
    Dataset Bulider. In essense a wrapped partial function.

    Example:
        >>> import matchzoo as mz
        >>> builder = mz.dataloader.DatasetBuilder(
        ...     mode='point'
        ... )
        >>> data = mz.datasets.toy.load_data()
        >>> gen = builder.build(data)
        >>> type(gen)
        <class 'matchzoo.dataloader.dataset.Dataset'>

    """

    def __init__(self, **kwargs):
        """Init."""
        self._kwargs = kwargs

    def build(self, data_pack, **kwargs) -> Dataset:
        """
        Build a Dataset.

        :param data_pack: DataPack to build upon.
        :param kwargs: Additional keyword arguments to override the keyword
            arguments passed in `__init__`.
        """
        return mz.dataloader.Dataset(
            data_pack, **{**self._kwargs, **kwargs}
        )


class DatasetBuilderV2(DatasetBuilder):
    """
    Dataset Bulider V2. In essense a wrapped partial function.

    Example:
        >>> import matchzoo as mz
        >>> builder = mz.dataloader.DatasetBuilderV2(
        ...     mode='point'
        ... )
        >>> data = mz.datasets.toy.load_data()
        >>> gen = builder.build(data)
        >>> type(gen)
        <class 'matchzoo.dataloader.dataset.DatasetV2'>
    """

    def build(self, data_pack, **kwargs) -> Dataset:
        """
        Build a Dataset.

        :param data_pack: DataPack to build upon.
        :param kwargs: Additional keyword arguments to override the keyword
            arguments passed in `__init__`.
        """
        return mz.dataloader.DatasetV2(
            data_pack, **{**self._kwargs, **kwargs}
        )
