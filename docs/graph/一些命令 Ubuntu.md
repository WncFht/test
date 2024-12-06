---
categories: graph
date: 2024-11-18T20:58:22+0800
dir: graph
modify: 2024-12-06T00:14:15+0800
share: true
tags:
  - graph
title: 一些命令 Ubuntu
---

```text
su -
visudo /etc/sudoers
username ALL=(ALL:ALL) NOPASSWD: ALL
```

以下是根据您提供的内容，整理成 Markdown 格式的博客文章：

---

# ubuntu 下修改主机名、用户名以及用户密码

## 引言

在刚开始接触 Linux 系统时，我们通常会安装 Ubuntu 系统（或者 CentOS）的虚拟机。在安装过程中，可能没有决定好主机名或用户名，或者使用了默认的名字。随着对系统的熟悉，可能会觉得默认的名称不够满意，这时就会想要修改主机名和用户名。很多人可能会考虑重新安装系统，但这既耗时又需要迁移数据。幸运的是，Linux 系统提供了修改主机名、用户名和用户密码的方法，本文将详细介绍在 Ubuntu 下如何进行这些修改，以解决您的困扰。

**注：** 为了保证修改顺利完成，或者出错了也可以修改，请多开几个 Terminal 终端，并都切换到 root 账户下。

```bash
sudo su
```

**切记切换到 root 用户下在进行操作。**

## 修改主机名

修改主机名，也就是修改 Terminal 上，提示文字的@后面的对应的名字，可以通过“hostname”命令查看主机名。

### 1. 修改 hostname 文件

使用 vim 编辑器打开 hostname 文件，也可以使用 gedit 文本编辑器打开。

```bash
sudo vi /etc/hostname
sudo gedit /etc/hostname
```

也可以通过 hostnamectl 来修改主机名：

```bash
hostnamectl set-hostname master
```

### 2. 修改 hosts 文件

修改 hosts 文件：

```bash
sudo vi /etc/hosts
sudo gedit /etc/hosts
```

完成上面两个步骤后重启机器可以看到主机名已经为修改后的主机名。

## 修改用户名

**切记切换到 root 用户下进行修改，普通用户下修改用户名后，执行 sudo 命令会提示密码错误。**

可以使用 sed 命令进行批量修改：

```bash
sed -i "s/\b<srcStr>\b/<desStr>/g" `grep <srcStr> -rl <filename>`
```

### 1. 修改 passwd 文件

将 passwd 中原用户名修改成新用户名：

```bash
vi /etc/passwd
```

或者使用如下命令修改：

```bash
sed -i "s/\bmaster\b/andy/g" `grep master -rl /etc/passwd`
```

### 2. 修改 shadow 文件

将 shadow 中原用户名修改成新用户名：

```bash
vi /etc/shadow
```

或者用如下命令修改：

```bash
sed -i "s/\bmaster\b/andy/g" `grep master -rl /etc/shadow`
```

### 3. 修改 home 目录下文件夹名

将 home 目录下用户文件夹名修改为新用户的名：

```bash
mv /home/master/ /home/andy
```

### 4. 修改 sudo 权限

建议使用方法 1，即修改用户组的方式，两种都执行也可以。

#### 方法 1：修改 group 用户组

修改 group 文件，将原来的用户名替换成新用户名。

```bash
vi /etc/group
```

或者用如下命令修改：

```bash
sed -i "s/\bmaster\b/andy/g" `grep master -rl /etc/group`
```

#### 方法 2：修改 sudoers 文件

将 sudoers 文件中原用户名替换成新用户名。如果没有，则可以直接添加新用户名。

```bash
Andy      ALL=(ALL:ALL) ALL             # 用户andy需要输入密码执行sudo命令
%andy     ALL=(ALL) ALL                 # 用户组andy里的用户输入密码执行sudo命令

andy ALL=(ALL) NOPASSWD: ALL            # 用户andy免密执行sudo命令
%andy ALL=(ALL) NOPASSWD: ALL           # 用户组里的用户andy免密执行sudo命令
```

### 5. 重启机器

执行完上述步骤后重启机器，即可以新用户名登录。这里没有修改密码，密码是原用户的密码。

## 修改用户密码

```bash
sudo passwd username          # 修改用户密码
sudo passwd root              # 修改root密码
```