#!/usr/bin/env python3
"""Download official remote-sensing archives without replacing existing files."""

import argparse
import hashlib
import re
from pathlib import Path

import requests

ROOT = Path('data/remote_sensing/raw')
LOVEDA = {
    'Train.zip': (4021669263, 'de2b196043ed9b4af1690b3f9a7d558f'),
    'Val.zip': (2425958254, '84cae2577468ff0b5386758bb386d31d'),
    'Test.zip': (3126023212, 'a489be0090465e01fb067795d24e6b47'),
}
POTSDAM_SIZE = 13324219686
POTSDAM_SHARE = 'https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/'


def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(session, url, target, expected_size, expected_md5=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != expected_size or (expected_md5 and md5(target) != expected_md5):
            raise RuntimeError(f'Existing file does not match official archive; refusing to replace: {target}')
        print(f'Already verified: {target}', flush=True)
        return
    partial = target.with_name(target.name + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError(f'Partial file is larger than expected: {partial}')
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            raise RuntimeError(f'Server did not honor Range request for {partial}')
        if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError(f'Unexpected Content-Range for {partial}')
        with partial.open('ab' if offset else 'wb') as stream:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    stream.write(block)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f'Incomplete download: {partial} ({partial.stat().st_size}/{expected_size})')
    if expected_md5 and md5(partial) != expected_md5:
        raise RuntimeError(f'MD5 mismatch: {partial}')
    partial.rename(target)
    print(f'Verified: {target}', flush=True)


def potsdam_session(password):
    session = requests.Session()
    page = session.get(POTSDAM_SHARE, timeout=30)
    page.raise_for_status()
    match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)', page.text)
    if not match:
        raise RuntimeError('Could not find ISPRS Seafile access form')
    page = session.post(
        POTSDAM_SHARE,
        data={'csrfmiddlewaretoken': match.group(1), 'token': '429be50cc79d423ab6c4',
              'password': password},
        headers={'Referer': POTSDAM_SHARE}, timeout=30)
    page.raise_for_status()
    if 'Potsdam.zip' not in page.text or 'File preview unsupported' not in page.text:
        raise RuntimeError('ISPRS Seafile did not accept the public access password')
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['loveda', 'potsdam', 'all'], default='all')
    parser.add_argument('--loveda-split', choices=['Train', 'Val', 'Test'],
                        help='Download only one LoveDA split')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--potsdam-password', default='CjwcipT4-P8g',
                        help='Public password shown on the ISPRS benchmark download page')
    args = parser.parse_args()
    if args.dataset in ('loveda', 'all'):
        with requests.Session() as session:
            files = ([(f'{args.loveda_split}.zip', LOVEDA[f'{args.loveda_split}.zip'])]
                     if args.loveda_split else LOVEDA.items())
            for name, (size, checksum) in files:
                url = f'https://zenodo.org/api/records/5706578/files/{name}/content'
                download(session, url, args.root / 'loveda' / name, size, checksum)
    if args.dataset in ('potsdam', 'all'):
        with potsdam_session(args.potsdam_password) as session:
            url = 'https://seafile.projekt.uni-hannover.de/seafhttp/f/429be50cc79d423ab6c4/'
            download(session, url, args.root / 'potsdam' / 'Potsdam.zip', POTSDAM_SIZE)


if __name__ == '__main__':
    main()
