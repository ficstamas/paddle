from typing import Optional, Literal

import tqdm

from paddle.utils.network import download_url
import os
from paddle.datasets.dataclasses import DataSplitsNYTKNerKor, NYTKNerKor
from paddle.datasets.utils.parser.conll import parse_conllu_plus
from paddle.datasets.utils.dataframe import concat_dataframes
import zipfile
import pandas as pd


_RESOURCE_URL = 'https://github.com/nytud/NYTK-NerKor/'
_DOWNLOAD_URL = 'https://github.com/nytud/NYTK-NerKor/archive/refs/heads/main.zip'
_CITATION = 'Simon, Eszter; Vadász, Noémi. (2021) Introducing NYTK-NerKor, ' \
            'A Gold Standard Hungarian Named Entity Annotated Corpus. ' \
            'In: Ekštein K., Pártl F., Konopík M. (eds) Text, Speech, and Dialogue. TSD 2021. ' \
            'Lecture Notes in Computer Science, vol 12848. Springer, Cham. https://doi.org/10.1007/978-3-030-83527-9_19'

_MORPH = ["news", "web", "wikipedia"]
_NO_MORPH = ["fiction", "legal", "news", "wikipedia"]

_DATA_PATH = "nytk_nerkor/NYTK-NerKor-main/data/genres/"


def download(path: Optional[str],
             retries: Optional[int] = 5,
             verify_ssl: Optional[bool] = True,
             regex: Optional[str] = None) -> Optional[str]:
    """
    Downloads Resource

    :param path: Destination
    :param retries: Maximum number of retries to acquire the resource
    :param verify_ssl: Verify SSL certificates
    :param regex: NOT USED
    :return: Path to the resource, or None
    """
    print("Dataset: ", _CITATION)

    if path.endswith(".zip"):
        path, filename = os.path.split(path)
    else:
        filename = "nytk_nerkor.zip"

    file = os.path.join(path, filename)
    if not os.path.exists(file):
        download_url(_DOWNLOAD_URL, file, retries, verify_ssl, {'desc': filename}, "get")

    return file


def _concat_paths(p_, files_):
    return [os.path.join(p_, f_) for f_ in files_]


def parse_file(path: str) -> pd.DataFrame:
    return parse_conllu_plus(file=path)


def load_dataset(path: str,
                 genre: Literal["all", "fiction", "legal", "news", "web", "wikipedia"] = "all",
                 morph: Optional[bool] = None,
                 download_if_necessary: Optional[bool] = True,
                 data_split: Optional[list] = (0.7, 0.1, 0.2),
                 random_state: Optional[int] = 42) -> DataSplitsNYTKNerKor:
    """
    Loads dataset

    :param path: Path to the resource folder
    :param genre: Selected genre
    :param morph: With or without morphological annotation, if set to None both will be returned
    :param download_if_necessary: Downloads the dataset if it can not found in the provided location
    :param data_split: size of the splits [train, dev, test], if None everything is going to be in the train split
    :param random_state: random state to use for the splits
    :return: Returns the lists of documents which has been split into lines
    """
    if data_split is not None and sum(data_split) != 1:
        raise ValueError("data_split must sum to 1")

    if download_if_necessary:
        path = download(path)

    # extracting files
    with zipfile.ZipFile(path) as f:
        f: zipfile.ZipFile
        path, filename = os.path.split(path)
        save_folder = os.path.join(path, filename.rstrip(".zip"))
        os.makedirs(save_folder, exist_ok=True)
        f.extractall(save_folder)

    # resolving input
    if genre == "all":
        genre_ = set(_MORPH + _NO_MORPH)
        if morph is None:
            morph_ = ["morph", "no-morph"]
        else:
            morph_ = ["morph"] if morph else ["no-morph"]
    else:
        if morph is not None:
            if morph and genre not in _MORPH:
                raise ValueError("No morphological annotation found for this genre!")
            elif not morph and genre not in _NO_MORPH:
                raise ValueError("This genre does not have a corpus without morphological annotation")
            else:
                morph_ = ["morph"] if morph else ["no-morph"]
        else:
            morph_ = ["morph", "no-morph"]
        genre_ = [genre, ]

    data_ = []

    # collecting data
    for g_ in tqdm.tqdm(genre_, 'Processing'):
        for m_ in morph_:
            p_ = os.path.join(os.path.join(_DATA_PATH, g_), m_)
            if not os.path.exists(p_):
                continue
            files_ = os.listdir(p_)
            paths_ = _concat_paths(p_, files_)
            dfs_ = [parse_file(conll_file) for conll_file in paths_]
            df = concat_dataframes(dfs_, "document_idx_")
            data_.append(NYTKNerKor(genre=g_, morph=m_ == "morph", data=df))

    if data_split is None:
        return DataSplitsNYTKNerKor(train=data_, dev=None, test=None)

    # Creating splits
    train = []
    dev = []
    test = []

    for doc_ in tqdm.tqdm(data_, desc='Creating Splits'):
        df = doc_.data
        train_ = df.sample(frac=data_split[0], random_state=random_state)
        dev_ = df.drop(train_.index).sample(frac=1 / (1 - data_split[0]) * data_split[1], random_state=random_state)
        test_ = df.drop(train_.index).drop(dev_.index)

        train.append(NYTKNerKor(genre=doc_.genre, morph=doc_.morph, data=train_))
        dev.append(NYTKNerKor(genre=doc_.genre, morph=doc_.morph, data=dev_))
        test.append(NYTKNerKor(genre=doc_.genre, morph=doc_.morph, data=test_))

    return DataSplitsNYTKNerKor(train=train, dev=dev, test=test)


dataset = load_dataset("./")
breakpoint()