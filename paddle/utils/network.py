import requests
import os
from typing import Optional
import tqdm


def download_url(url: Optional[str],
                 output: Optional[str],
                 retries: Optional[int] = 5,
                 verify_ssl: Optional[bool] = True) -> Optional[str]:
    """
    Downloads file from the specified URL to the output folder

    :param url: URL
    :param output: Output folder
    :param retries: Maximum number of retries to acquire the resource
    :param verify_ssl: Verify SSL certificates
    :return: Path to the resource, or None
    """

    path, filename = os.path.split(output)
    os.makedirs(path, exist_ok=True)

    while retries + 1 > 0:
        try:
            print(f'Downloading {filename} from {url} ...')
            resp = requests.get(url, stream=True, verify=verify_ssl)
            if resp.status_code != 200:
                raise RuntimeError(f'Failed downloading url {url}')

            total_size = int(resp.headers.get('content-length', 0))
            chunk_size = 1024
            with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True) as t:
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

