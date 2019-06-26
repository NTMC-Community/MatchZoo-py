from matchzoo.engine.base_callback import BaseCallback


class LambdaCallback(BaseCallback):
    """
    LambdaCallback. Just a shorthand for creating a callback class.

    See :class:`matchzoo.engine.base_callback.BaseCallback` for more details.

    Example:

        >>> import matchzoo as mz
        >>> from matchzoo.dataloader.callbacks import LambdaCallback
        >>> data = mz.datasets.toy.load_data()
        >>> batch_func = lambda x: print(type(x))
        >>> unpack_func = lambda x, y: print(type(x), type(y))
        >>> callback = LambdaCallback(on_batch_data_pack=batch_func,
        ...                           on_batch_unpacked=unpack_func)
        >>> dataset = mz.dataloader.Dataset(
        ...     data, callbacks=[callback])
        >>> _ = dataset[0]
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        <class 'dict'> <class 'numpy.ndarray'>

    """

    def __init__(self, on_batch_data_pack=None, on_batch_unpacked=None):
        """Init."""
        self._on_batch_unpacked = on_batch_unpacked
        self._on_batch_data_pack = on_batch_data_pack

    def on_batch_data_pack(self, data_pack):
        """`on_batch_data_pack`."""
        if self._on_batch_data_pack:
            self._on_batch_data_pack(data_pack)

    def on_batch_unpacked(self, x, y):
        """`on_batch_unpacked`."""
        if self._on_batch_unpacked:
            self._on_batch_unpacked(x, y)
