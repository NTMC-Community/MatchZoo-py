"""Download file."""
import os
import hashlib
import shutil
import sys
import tarfile
import time
import zipfile
import collections
from contextlib import closing
import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen
import numpy as np

import matchzoo


class Progbar(object):
    """
    Displays a progress bar.

    :param target: Total number of steps expected, None if unknown.
    :param width: Progress bar width on screen.
    :param verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
    :param stateful_metrics: Iterable of string names of metrics that
        should *not* be averaged over time. Metrics in this list
        will be displayed as-is. All others will be averaged
        by the progbar before display.
    :param interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(
        self,
        target,
        width=30,
        verbose=1,
        interval=0.05,
        stateful_metrics=None
    ):
        """Init."""
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout,
                                  'isatty') and sys.stdout.isatty()
                                  ) or 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar."""
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and self.target is not
               None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        """
        Replacement for `urlretrive` for Python 2.
        Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
        `urllib` module, known to have issues with proxy management.

        :param url: url to retrieve.
        :param filename: where to store the retrieved data locally.
        :param reporthook: a hook function that will be called once
            on establishment of the network connection and once
            after each block read thereafter.
            The hook will be passed three arguments;
            a count of blocks transferred so far,
            a block size in bytes, and the total size of the file.
        :param data: `data` argument passed to `urlopen`.
        """
        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while True:
                chunk = response.read(chunk_size)
                count += 1
                if reporthook is not None:
                    reporthook(count, chunk_size, total_size)
                if chunk:
                    yield chunk
                else:
                    break

        with closing(urlopen(url, data)) as res, open(filename, 'wb') as fd:
            for chunk in chunk_read(res, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def _extract_archive(file_path, path='.', archive_format='auto'):
    """
    Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    :param file_path: path to the archive file
    :param path: path to extract the archive file
    :param archive_format: Archive format to try for extracting the file.
        Options are 'auto', 'tar', 'zip', and None.
        'tar' includes tar, tar.gz, and tar.bz files.
        The default 'auto' is ['tar', 'zip'].
        None or an empty list will return no matches found.

    :return: True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError,
                        KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='data',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=matchzoo.USER_DATA_DIR):
    """
    Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.matchzoo/datasets`, placed in the cache_subdir `data`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.matchzoo/datasets/data/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    :param fname: Name of the file. If an absolute path `/path/to/file.txt` is
        specified the file will be saved at that location.
    :param origin: Original URL of the file.
    :param untar: Deprecated in favor of 'extract'. Boolean, whether the file
        should be decompressed.
    :param md5_hash: Deprecated in favor of 'file_hash'. md5 hash of the file
        for verification
    :param file_hash: The expected hash string of the file after download.
        The sha256 and md5 hash algorithms are both supported.
    :param cache_subdir: Subdirectory under the cache dir where the file is
        saved. If an absolute path `/path/to/folder` is specified the file
        will be saved at that location.
    :param hash_algorithm: Select the hash algorithm to verify the file.
        options are 'md5', 'sha256', and 'auto'. The default 'auto' detects
        the hash algorithm in use.
    :papram extract: True tries extracting the file as an Archive, like tar
        or zip.
    :param archive_format: Archive format to try for extracting the file.
        Options are 'auto', 'tar', 'zip', and None.
        'tar' includes tar, tar.gz, and tar.bz files.
        The default 'auto' is ['tar', 'zip'].
        None or an empty list will return no matches found.
    :param cache_dir: Location to store cached files, when None it defaults to
        the [matchzoo.USER_DATA_DIR](~/.matchzoo/datasets).

    :return: Path to the downloaded file.
    """
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.matchzoo')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the file hash '
                      'does not match the original value of file_hash.'
                      ' We will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        class ProgressTracker(object):
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {} : {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """
    Validates a file against a sha256 or md5 hash.

    :param fpath: path to the file being validated
    :param file_hash:  The expected hash string of the file.
        The sha256 and md5 hash algorithms are both supported.
    :param algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
        The default 'auto' detects the hash algorithm in use.
    :param chunk_size: Bytes to read at a time, important for large files.

    :return: Whether the file is valid.
    """
    if ((algorithm == 'sha256') or (algorithm == 'auto' and len(
                                    file_hash) == 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """
    Calculates a file sha256 or md5 hash.

    :param fpath: path to the file being validated
    :param algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
        The default 'auto' detects the hash algorithm in use.
    :param chunk_size: Bytes to read at a time, important for large files.

    :return: The file hash.
    """
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()
