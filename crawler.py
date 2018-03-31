# -*- coding: utf-8 -*-
import os
import json
import queue
import datetime
import threading
import socket
from urllib import request
from urllib.parse import urlencode
from urllib import error
from http import client
# 下载线程数量
thread_num = 10
# temporary API KEY
api_key = 'a0a4dea38c341f52543ef6be07d85281'
# 本地存储路径
path = 'photos/'
# url地址
base_url = 'https://api.flickr.com/services/rest/'
# 参数(不簽署呼叫)
parameters = { 
    'method' : 'flickr.interestingness.getList',
    'api_key' : api_key,
    'format' : 'json',
    'nojsoncallback' : '1',
    'per_page' : '500',
    'date' : datetime.date.today()
}
def find_url(date):
    print(date)
    parameters['date'] = date
    # 进行参数封装
    data = urlencode(parameters)
    # 组装完整url
    url_visit = base_url + '?' + data
    #print(url_visit)
    # 访问完整url
    try:
        content = request.urlopen(url_visit,timeout=5).read()
    except client.IncompleteRead as e:
        raise
        return
    except error.HTTPError as e:
        raise
        return
    except socket.timeout as e:
        raise
        return
    except error.URLError as e:
        raise
        return
    #print(content.decode('unicode-escape'))
    # 解析JSON
    json_content = json.loads(content.decode('utf8'))
    if json_content['stat'] == 'ok':
        for photo in json_content['photos']['photo']:
            # 图片URL
            photo_url = 'https://farm{farm}.staticflickr.com/{server}/{id}_{secret}.jpg'.format(farm=photo['farm'],server=photo['server'],id=photo['id'],secret=photo['secret'])
            que.put({"id": photo['id'], "url": photo_url})
    # 启动多线程进行下载
    for i in range(thread_num):
        d = photo_downloader(que)
        d.start()
            
class photo_downloader(threading.Thread):
    def __init__(self,que):
        threading.Thread.__init__(self)
        self.que = que
    def run(self):
        while not que.empty():
            photo = self.que.get()
            photo_url = photo['url']
            # 下载图片
            try:
                request.urlretrieve(photo_url,path+photo['id']+'.jpg')
            # 删除下载失败的图片
            except error.ContentTooShortError as e:
                os.remove(path+photo['id']+'.jpg')
                continue
            except error.URLError as e:
                continue
            #except client.RemoteDisconnected as e:
             #   continue
            photo_data = open(path+photo['id']+'.jpg','rb').read()
            # 根据文件头删除格式错误的图片
            if(photo_data[:2]!=b'\xff\xd8'):
                os.remove(path+photo['id']+'.jpg')
            

if __name__ == '__main__':
    que = queue.Queue()
    #date = datetime.date.today()
    date = datetime.date(2017,5,26)
    while True:
        # 总共需要的图片数量（会略多一些）
        if(len([x for x in os.listdir(path)])<100000):
            date -= datetime.timedelta(1)
            # 可能由于网络不好等原因报异常，这种情况下date不变，重新调用find_url
            try:
                find_url(date)
            except client.IncompleteRead as e:
                date += datetime.timedelta(1)
            except error.HTTPError as e:
                date += datetime.timedelta(1)
            except socket.timeout as e:
                date += datetime.timedelta(1)
            except error.URLError as e:
                date += datetime.timedelta(1)
        # 主线程查询当前正在活动的线程数量，当数量为1的时候，即只剩主线程的时候，表示队列中所有图片下载完毕，更新date再次获取一系列图片的URL
            while(threading.active_count()>1):
                pass
        else:
            break
            
    
        
    
    






