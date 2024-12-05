---
title: pip 相关
date: 2024-10-31T20:19:02+0800
modify: 2024-12-06T00:10:44+0800
categories: Computer
dir: Obsidian
share: true
tags:
  - Computer
  - Technology
---

```~/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

```shell
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

```
mkdir -p ~/.pip
echo "[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.pip/pip.conf
```

```
mkdir %APPDATA%\pip
echo "[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple" > %APPDATA%\pip\pip.ini
```

```shell
pip freeze > requirements.txt
```