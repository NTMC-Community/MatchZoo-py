"""Average meter."""


class AverageMeter(object):
    """
    Computes and stores the average and current value.

    Examples:
        >>> am = AverageMeter()
        >>> am.update(1)
        >>> am.avg
        1.0
        >>> am.update(val=2.5, n=2)
        >>> am.avg
        2.0

    """

    def __init__(self):
        """Average meter constructor."""
        self.reset()

    def reset(self):
        """Reset AverageMeter."""
        self._val = 0.
        self._avg = 0.
        self._sum = 0.
        self._count = 0.

    def update(self, val, n=1):
        """Update value."""
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count

    @property
    def avg(self):
        """Get avg."""
        return self._avg
