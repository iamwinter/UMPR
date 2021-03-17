import argparse
import gzip
import json
import os
import socket
import sys
import time
import urllib
import uuid
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
from urllib.request import urlretrieve

os.chdir(sys.path[0])

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
            # print(f'#### Downloaded {os.path.basename(path)} {url}')
            return True
        except Exception:  # 爬取图片失败，短暂sleep后重新爬取
            # print(f'#### Download failed. Reload: {url}')
            time.sleep(0.5)
    print(f'#### Download failed: {os.path.basename(path)} {url}')
    return False


def download_photos(meta_path, max_workers=256):
    data_dir = os.path.dirname(meta_path)
    photo_dir = os.path.join(data_dir, 'photos')
    os.makedirs(photo_dir, exist_ok=True)

    if not os.path.exists(os.path.join(data_dir, 'photos.json')):
        print('#### Read item list from "items.csv"!')
        items_set = set()
        try:
            items_df = pd.read_csv(os.path.join(data_dir, 'items.csv'))  # Read list which shows existing items.
            items_set = set(items_df['itemID'])
        except Exception:
            print('#### Please first running "data_process.py" to generate "items.csv"!!!')
            exit(-1)

        print('#### Read the meta file to create photos.json')
        if meta_path.endswith('.gz'):
            f = gzip.open(meta_path, 'rb')
        else:
            f = open(meta_path, 'r', encoding='UTF-8')
        photos = []
        for line in f.readlines():
            item = json.dumps(eval(line))
            item = json.loads(item)
            if 'imUrl' in item and item['asin'] in items_set:  # Only download pictures of existing items.
                photo_name = uuid.uuid4().hex[:16]
                photos.append([item['asin'], photo_name, item['imUrl']])
        f.close()
        df = pd.DataFrame(photos, columns=['business_id', 'photo_id', 'imUrl'])
        df.to_json(os.path.join(data_dir, 'photos.json'))
        print(f'#### photos.json has been saved which contains {len(photos)} pictures.')
    else:
        print(f'#### Read the existing file photos.json.')
        df = pd.read_json(os.path.join(data_dir, 'photos.json'))

    print(f'#### Start to download pictures and save them into {photo_dir}')
    pool = ThreadPoolExecutor(max_workers=max_workers)
    tasks = []
    for name, url in zip(df['photo_id'], df['imUrl']):
        path = os.path.join(photo_dir, name+'.jpg')
        if not os.path.exists(path) or not is_valid_jpg(path):
            task = pool.submit(download_photo, url, path)
            tasks.append(task)

    success = 0
    for i, task in enumerate(as_completed(tasks)):
        success += task.result()
        print(f'#### Tried {i}/{len(tasks)} photos!', end='\r', flush=True)
    print(f'#### {success} images downloaded successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_path', dest='meta_path', default='./music_small/meta_Digital_Music.json.gz')
    args = parser.parse_args()

    download_photos(args.meta_path)
    print(f'## "down_amazon_photos.py" Completed!')
