---
title: ROS2-踩坑
date: 2024-10-18T00:15:58+0800
modify: 2024-12-06T00:13:52+0800
categories: graph
dir: Obsidian
share: true
tags:
  - graph
---

```shell
 fht@fht-System-Product-Name  ~  cd /opt/ros/  
 fht@fht-System-Product-Name  /opt/ros  ls
humble
 fht@fht-System-Product-Name  /opt/ros  cd humble   
 fht@fht-System-Product-Name  /opt/ros/humble  ls
bin    include  local             local_setup.sh        local_setup.zsh  setup.bash  setup.zsh  src
cmake  lib      local_setup.bash  _local_setup_util.py  opt              setup.sh    share      tools
 fht@fht-System-Product-Name  /opt/ros/humble  setup.zsh          
zsh: command not found: setup.zsh
 ✘ fht@fht-System-Product-Name  /opt/ros/humble  zsh setup.zsh        
 fht@fht-System-Product-Name  /opt/ros/humble  ROS --version 
zsh: command not found: ROS
 ✘ fht@fht-System-Product-Name  /opt/ros/humble  printenv | grep -i ROS
OLDPWD=/opt/ros
PWD=/opt/ros/humble
 fht@fht-System-Product-Name  /opt/ros/humble  source setup.zsh                      
 fht@fht-System-Product-Name  /opt/ros/humble  printenv | grep -i ROS
OLDPWD=/opt/ros
PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
PWD=/opt/ros/humble
AMENT_PREFIX_PATH=/opt/ros/humble
PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages
LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib
ROS_DISTRO=humble
ROS_LOCALHOST_ONLY=0
ROS_PYTHON_VERSION=3
ROS_VERSION=2
 fht@fht-System-Product-Name  /opt/ros/humble  ros2 --version
usage: ros2 [-h] [--use-python-default-buffering] Call `ros2 <command> -h` for more detailed usage. ...
ros2: error: unrecognized arguments: --version
 ✘ fht@fht-System-Product-Name  /opt/ros/humble  
```

记得在 `~/.zshrc` 中加上

```shell
source /opt/ros/humble/setup.zsh
```