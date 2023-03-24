import sys

import you_get


def download(url, path):
    sys.argv = ['you-get', '-o', path, url]

    you_get.main()


if __name__ == '__main__':
    # 视频网站的地址
    url = 'https://www.bilibili.com/bangumi/play/ep118488?from=search&seid=5050973611974373611'
    # 视频输出的位置
    path = 'D:\\Users\\HP\\Desktop\\project\\detect\\video'
    download(url, path)

# you-get -o D:\\Users\\HP\\Desktop\\project\\detect\\video\\ -format=flv480  https://www.bilibili.com/video/BV1Y24y127zB/?spm_id_from=333.337.search-card.all.click
