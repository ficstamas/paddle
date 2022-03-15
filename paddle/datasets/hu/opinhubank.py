from typing import Optional
from paddle.utils.network import download_url
import os
import pandas as pd
import numpy as np
from paddle.datasets.dataclasses import DataSplitsOpinHuBank
import zipfile


_RESOURCE_URL = 'http://metashare.nytud.hu/repository/download/608756be64e211e2aa7c68b599c26a068dd5b3551f024f6281131670412d37d3/'
_CITATION = 'Miháltz, Márton (2013). “OpinHuBank: szabadon hozzáférhető annotált korpusz magyar nyelvű véleményelemzéshez”. ' \
            'Tanács Attila, Vincze Veronika (szerk.): IX. Magyar Számítógépes Nyelvészeti Konferencia (MSZNY 2013), SZTE, Szeged, 2013, pp. 343-345.'


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
        filename = "opinhubank.zip"

    file = os.path.join(path, filename)
    if not os.path.exists(file):
        download_url(_RESOURCE_URL, file, retries, verify_ssl, {'desc': filename}, "post", {
            "licence_agree": "on",
            "in_licence_agree_form": "True",
            "licence": "CC-BY"
        })

    return file


def load_dataset(path: Optional[str],
                 download_if_necessary: Optional[bool] = True,
                 regex: Optional[str] = None,
                 data_split: Optional[list] = (0.7, 0.1, 0.2),
                 random_state: Optional[int] = 42) -> DataSplitsOpinHuBank:
    """
    Loads dataset

    :param path: Path to the resource folder
    :param download_if_necessary: Downloads the dataset if it can not found in the provided location
    :param regex: NOT USED
    :param data_split: size of the splits [train, dev, test], if None everything is going to be in the train split
    :param random_state: random state to use for the splits
    :return: Returns the lists of documents which has been split into lines
    """

    if sum(data_split) != 1:
        raise ValueError("data_split must sum to 1")

    if download_if_necessary:
        path = download(path, regex=regex)

    output = []

    with zipfile.ZipFile(path) as f:
        f: zipfile.ZipFile
        path, filename = os.path.split(path)
        save_folder = os.path.join(path, filename.rstrip(".zip"))
        os.makedirs(save_folder, exist_ok=True)
        f.extractall(save_folder)

    file = [file for file in os.listdir(save_folder) if file.endswith(".csv")][0]

    df = pd.read_csv(os.path.join(save_folder, file), encoding='iso-8859-2')

    if data_split is None:
        return DataSplitsOpinHuBank(train=df, test=None, dev=None)

    train = df.sample(frac=data_split[0], random_state=random_state)
    dev = df.drop(train.index).sample(frac=1/(1-data_split[0])*data_split[1], random_state=random_state)
    test = df.drop(train.index).drop(dev.index)

    return DataSplitsOpinHuBank(train=train, test=test, dev=dev)
