from  gevent import monkey
monkey.patch_all()          # python 网络接口多进程处理包
import multiprocessing

debug = False
bind = "0.0.0.0:8050"       # 绑定的ip及端口号
backlog=2048                # 监听队列
threads=1                   # 每个进程的线程数
worker_connections=3000     # 最大并发量
workers = 5
worker_class = "gevent"
timeout=600