---
author: xianpeng zhao
date: 2018年8月6日星期一 上午11:54
title: Tips
---

## 怎样通过ssh tunnel的方式翻墙

1. 先在墙上打个洞

   ```
    ssh -fND localhost:8888 root@remote_ip # 透過 ssh 建立一個 socks5 通過代理上網
   ```

   例如，要利用我们美国的server建立ssh tunnel只需要执行：

   ssh -fND localhost:1080 xpzhao@hq-dev2.aerohive.com   

2. 然后本机想通过该tunnel上网的时候，只需要把代理配置到洞的这一边，数据自然就会从洞中过了。

   1. 如果是浏览器想翻墙的话，配置sock5代理到127.0.0.1:1080即可

   2. 如果是shell想要上网的话，由于终端工具可能没有配置代理的接口，需要借助其他软件，常用的软件为proxychains来实现代理功能代理到127.0.0.1:1080即可。

      