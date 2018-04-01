import requests
from bs4 import BeautifulSoup
import pickle
import time

baseurl = 'https://sanskritbible.in/assets/php/read-btxt.php'
book_no = 65
no_chapters = 1
params = {
    'BookNo':
    book_no,
    'User-Agent':
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:58.0) Gecko/20100101 Firefox/58.0',
    'X-Requested-With':
    'XMLHttpRequest',
    'Host':
    'www.sanskritbible.in'
}

h = 0
t = 0
u = 0
l = []
try:
    for i in range(no_chapters):
        u += 1
        if u > 9:
            u = 0
            t += 1
            if t > 9:
                t = 0
                h += 1

        params['ChapterNo'] = '{}{}{}'.format(h, t, u)
        r = requests.post(baseurl, json=params)
        try:
            j = r.json()
            s = j[0]
            e = j[4]
            assert len(s) == len(e)
            for i in range(len(s)):
                if 't' in s[i] and 't' in e[i]:
                    l.append((s[i]['t'], e[i]['t']))
        except AssertionError:
            continue
        except:
            break
    print('{}{}{}'.format(h, t, u))

finally:
    print('{}{}{}'.format(h, t, u))

    with open('pickles/{}_{}.pickle'.format(time.ctime(), len(l)), 'wb') as f:
        pickle.dump(l, f)
