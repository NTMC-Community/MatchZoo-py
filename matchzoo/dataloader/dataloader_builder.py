import matchzoo as mz
from matchzoo.dataloader import DataLoader


class DataLoaderBuilder(object):
    """
    DataLoader Bulider. In essense a wrapped partial function.

    Example:
        >>> import matchzoo as mz
        >>> padding_callback = mz.dataloader.callbacks.BasicPadding()
        >>> builder = mz.dataloader.DataLoaderBuilder(
        ...     stage='train', callback=padding_callback
        ... )
        >>> data_pack = mz.datasets.toy.load_data()
        >>> preprocessor = mz.preprocessors.BasicPreprocessor()
        >>> data_processed = preprocessor.fit_transform(data_pack)
        >>> dataset = mz.dataloader.Dataset(data_processed, mode='point')
        >>> dataloder = builder.build(dataset)
        >>> type(dataloder)
        <class 'matchzoo.dataloader.dataloader.DataLoader'>

    """

    def __init__(self, **kwargs):
        """Init."""
        self._kwargs = kwargs

    def build(self, dataset, **kwargs) -> DataLoader:
        """
        Build a DataLoader.

        :param dataset: Dataset to build upon.
        :param kwargs: Additional keyword arguments to override the keyword
            arguments passed in `__init__`.
        """
        return mz.dataloader.DataLoader(
            dataset, **{**self._kwargs, **kwargs}
        )
