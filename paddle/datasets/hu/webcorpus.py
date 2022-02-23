from typing import Optional, List
from paddle.utils.network import get_url_paths, download_url
import re
import os
import tqdm
from paddle.datasets.dataclasses import DataSplits


_RESOURCE_URL = 'https://nessie.ilab.sztaki.hu/~ndavid/Webcorpus2_text/'
_CITATION = 'Nemeskey, Dávid Márk (2020). “Natural Language Processing methods for Language Modeling”. ' \
            'PhD thesis. Eötvös Loránd University.'


def download(path: Optional[str],
             retries: Optional[int] = 5,
             verify_ssl: Optional[bool] = True,
             regex: Optional[str] = None) -> Optional[List]:
    """
    Downloads Resource

    :param path: Destination
    :param retries: Maximum number of retries to acquire the resource
    :param verify_ssl: Verify SSL certificates
    :param regex: Downloads files which match the provided regex e.g. 'wiki*'
    :return: Path to the resource, or None
    """
    print(_CITATION)
    paths = get_url_paths(_RESOURCE_URL, ext='txt.gz')
    if regex is not None:
        regex = re.compile(regex)
        paths = [p for p in paths if re.match(regex, p.split('/')[-1]) is not None]

    output = []

    for p in paths:
        filename = p.split('/')[-1]
        file = os.path.join(path, filename)
        output.append(file)
        if not os.path.exists(file):
            download_url(p, file, retries, verify_ssl, {'desc': filename})

    return output


def load_dataset(path: Optional[str],
                 download_if_necessary: Optional[bool] = True,
                 regex: Optional[str] = None) -> DataSplits:
    """
    Loads dataset

    :param path: Path to the resource folder
    :param download_if_necessary: Downloads the dataset if it can not found in the provided location
    :param regex: Downloads files which match the provided regex e.g. 'wiki*'
    :return: Returns the lists of documents which has been split into lines
    """

    if download_if_necessary:
        paths = download(path, regex=regex)
    else:
        paths = os.listdir(path)

    output = []
    for p in tqdm.tqdm(paths, desc='Loading'):
        with open(p, mode='r', encoding='utf8') as f:
            lines = f.readlines()
            output.append([l.strip('\n') for l in lines if len(l.strip('\n')) > 0])

    return DataSplits(train=output, test=None, dev=None)
