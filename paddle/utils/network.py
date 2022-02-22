import requests
import os
from typing import Optional, List
import tqdm
import requests
from bs4 import BeautifulSoup


def download_url(url: Optional[str],
                 output: Optional[str],
                 retries: Optional[int] = 5,
                 verify_ssl: Optional[bool] = True,
                 tqdm_params: Optional[dict] = None) -> Optional[str]:
    """
    Downloads file from the specified URL to the output folder

    :param url: URL
    :param output: Output folder
    :param retries: Maximum number of retries to acquire the resource
    :param verify_ssl: Verify SSL certificates
    :param tqdm_params: Parameters for tqdm customization
    :return: Path to the resource, or None
    """

    path, filename = os.path.split(output)
    os.makedirs(path, exist_ok=True)

    if tqdm_params is None:
        tqdm_params = {}

    while retries + 1 > 0:
        try:
            print(f'Downloading {filename} from {url} ...')
            resp = requests.get(url, stream=True, verify=verify_ssl)
            if resp.status_code != 200:
                raise RuntimeError(f'Failed downloading url {url}')

            total_size = int(resp.headers.get('content-length', 0))
            chunk_size = 1024
            with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True, **tqdm_params) as t:
                with open(output, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:  # filter out keep-alive new chunks
                            if tqdm is not None:
                                t.update(len(chunk))
                            f.write(chunk)
            break
        except Exception as e:
            retries -= 1
            if retries <= 0:
                raise e
            print(f'Download failed due to {repr(e)}, retrying, {retries} attempt left')
    return output


def get_url_paths(url, ext='', params=None) -> Optional[List]:
    """
    Retrieve file list in web folder

    :param url: URL
    :param ext: File Extension to filter files e.g. png, iso, zip...
    :param params: Additional parameters for the GET method
    :return: List of locations
    """
    if params is None:
        params = {}

    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent
