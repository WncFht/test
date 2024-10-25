## SSH 概述

SSH是一种加密的网络协议，用于安全地访问远程计算机。

## SSH 认证原理

- 口令认证(password)
    - 当尝试连接到目标主机时，会要求你输入当前主机上你要连接的用户的密码，输入成功即可成功连接

> [!warning]  
> 在生产环境中请不要开启ssh的口令登录

- 公私钥认证(publickey)
    - 在连接时，主机和你的电脑将会进行多次交互以验证你当前使用的私钥是否与其允许连接的公钥中（一般储存在`~/.ssh/authorized_keys`中）的某一个相匹配，如果匹配则连接成功

>[!info] 如何配置公私钥认证登录
>1. 生成一对公私钥
>2. 将公钥拷贝道 `~/.ssh/authorized_keys` 中  
>[SSH 密钥登录 菜鸟教程](https://www.cainiaojc.com/ssh/ssh-key.html)

## 中间人攻击

## 常用的 SSH 命令

 [SSH 教程 菜鸟教程](https://www.cainiaojc.com/ssh/ssh-index.html)

 
