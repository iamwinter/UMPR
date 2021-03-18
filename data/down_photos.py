import argparse
import os
import socket
import sys
import time
import urllib
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
from urllib.request import urlretrieve


socket.setdefaulttimeout(20)
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent',
                      'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36')]
urllib.request.install_opener(opener)


def is_valid_jpg(path):
    try:
        with open(path, 'rb') as f:
            f.seek(-2, 2)
            return f.read() == b'\xff\xd9'
    except Exception:
        return False


def download_photo(url, path):
    for epoch in range(10):  # 最多重新获取10次
        try:
            urlretrieve(url, path)
            return True, None, None
        except Exception:  # 爬取图片失败，短暂sleep后重新爬取
            time.sleep(0.5)
    return False, url, path


def download_photos(meta_path, max_workers=256):
    data_dir = os.path.dirname(meta_path)
    photo_dir = os.path.join(data_dir, 'photos')
    os.makedirs(photo_dir, exist_ok=True)

    try:
        print(f'## Read {meta_path}')
        df = pd.read_json(os.path.join(data_dir, 'photos.json'), orient='records', lines=True)
    except:
        print('## Please first running "data_process.py" to generate "photos.json"!!!')
        return

    print(f'## Start to download pictures and save them into {photo_dir}')
    pool = ThreadPoolExecutor(max_workers=max_workers)
    tasks = []
    for name, url in zip(df['photo_id'], df['imUrl']):
        path = os.path.join(photo_dir, name + '.jpg')
        if not os.path.exists(path) or not is_valid_jpg(path):
            task = pool.submit(download_photo, url, path)
            tasks.append(task)

    failed = []
    for i, task in enumerate(as_completed(tasks)):
        res, url, path = task.result()
        if not res:
            failed.append((url, path))
        print(f'## Tried {i}/{len(tasks)} photos!', end='\r', flush=True)

    for url, path in failed:
        print(f'## Failed to download {url} to {path}')
    print(f'## {len(tasks) - len(failed)} images were downloaded successfully to {photo_dir}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--photos_json', dest='photos_json', default=os.path.join(sys.path[0], 'music/photos.json'))
    args = parser.parse_args()

    download_photos(args.photos_json)
