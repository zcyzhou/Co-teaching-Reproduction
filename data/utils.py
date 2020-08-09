"""
Utils functions for the data processing
"""
import os
import os.path
import hashlib
import gzip
import errno
import tarfile
from typing import Any, Callable, List, Iterable, Optional, TypeVar
import zipfile

import torch
from torch.utils.model_zoo import tqdm

import numpy as np
from numpy.testing import assert_array_almost_equal


def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url(url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:   # download the file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def list_dir(root: str, prefix: bool = False) -> List[str]:
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root: str, suffix: str, prefix: bool = False) -> List[str]:
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def _quota_exceeded(response: "requests.models.Response") -> bool:  # type: ignore[name-defined]
    return "Google Drive - Quota exceeded" in response.text


def download_file_from_google_drive(file_id: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        if _quota_exceeded(response):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        _save_response_content(response, fpath)


def _get_confirm_token(response: "requests.models.Response") -> Optional[str]:  # type: ignore[name-defined]
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(
    response: "requests.models.Response", destination: str, chunk_size: int = 32768,  # type: ignore[name-defined]
) -> None:
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()


def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")


def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


T = TypeVar("T", str, bytes)


def verify_str_arg(
    value: T, arg: Optional[str] = None, valid_values: Iterable[T] = None, custom_msg: Optional[str] = None,
) -> T:
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ("Unknown value '{value}' for argument {arg}. "
                   "Valid values are {{{valid_values}}}.")
            msg = msg.format(value=value, arg=arg,
                             valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value

# Functions for adding noise to the dataset

# basic function
def multiclass_noisify(y, P, random_state=0):
    """Flip classes according to transition probability matrix T.
    It expects a number between 0 and #class-1.

    Args:
        y: matrix of train labels
        P: row stochastic matrix
        random_state: seed for the np.random
    """
    # np.max was used, but error occured:
    # max() received an invalid combination of arguments - got (out=NoneType, axis=NoneType, )
    # Solved:
    # np.max need to np.asarray first
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1] # Make sure P is square matrix
    assert np.max(y) < P.shape[0], "np.max(y) = {}, P.shape[0] = {}.".format(np.max(y), P.shape[0])

    # Check if P is a row stochastic matrix (The sum of each row == 1)
    # NOTE:
    #    In ndarray, axis = 0 means operate on axis 0
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print("Dim on axis 0 of y: {}".format(m))

    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        # The y is [[][][][]...[][]]
        # So i is an array with only one element
        i = y[idx]

        # [0] in the end is because the flipped will [[]] in default
        # The second parameter define the distribution of Pr
        flipped = flipper.multinomial(1, P[i][0], 1)[0]

        # Refer to the numpy doc, where() should return elements
        # However, when there's only condition specified, it will return index
        # new_y[idx] = np.where(flipped == 1)[0]
        new_y[idx] = np.asarray(flipped == 1).nonzero()[0]

    return new_y

# multiclass_noisify_pairflip call the function "multiclass_noisify"
def multiclass_noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """Mistakes:
        flip in pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # P[0, 0], P[0, 1] = 1.-n, n
        for i in range(0, nb_classes-1):
            P[i, i], P[i, i+1] = 1.-n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1.-n, n

        y_train_noisy = multiclass_noisify(y_train, P, random_state)
        actual_noise = (y_train_noisy != y_train).mean()

        assert actual_noise > 0.0, "Actual noise is 0."
        print("Actual noise: {:.2f}".format(actual_noise))
        # y_train = y_train_noisy
    print(P)

    return y_train_noisy, actual_noise

def multiclass_noisify_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """Mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes-1)) * P

    if n > 0.0:
        for i in range(0, nb_classes):
            P[i, i] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P, random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        print("Actual noise: {:.2f}".format(actual_noise))
        # y_train = y_train_noisy
    print(P)

    return y_train_noisy, actual_noise

def noisify(train_labels=None, noise_type=None, noise_rate=0, random_state=0, nb_classes=10):
    """Adding one of the two types of noise into the dataset.

    Return:
        [0] train_labels_noisy
        [1] actual_noise_rate
    """
    if noise_type == 'pairflip':
        train_labels_noisy, actual_noise_rate = multiclass_noisify_pairflip(train_labels, noise_rate, random_state, nb_classes)
    elif noise_type == 'symmetric':
        train_labels_noisy, actual_noise_rate = multiclass_noisify_symmetric(train_labels, noise_rate, random_state, nb_classes)
    return train_labels_noisy, actual_noise_rate
