---
categories: graph
date: 2024-11-21T09:02:44+0800
dir: graph
modify: 2024-12-06T00:13:57+0800
share: true
tags:
  - graph
title: Tabby + Zsh
---

# 指南 

## 一、前置准备

1. 系统要求

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    python3-pip \
    pkg-config \
    libssl-dev
```

2. ZSH 安装

```bash
# 安装zsh
sudo apt install zsh

# 设置为默认shell
chsh -s $(which zsh)

# 确认设置
echo $SHELL
# 应该输出: /usr/bin/zsh
```

## 二、ZSH 基础配置

### 1. Oh My Zsh 安装

```bash
# 安装Oh My Zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# 备份默认配置
cp ~/.zshrc ~/.zshrc.backup
```

### 2. 插件管理器安装

```bash
# 安装zinit
bash -c "$(curl --fail --show-error --silent --location https://raw.githubusercontent.com/zdharma-continuum/zinit/HEAD/scripts/install.sh)"

# 等待安装完成后重启终端
exec zsh
```

### 3. 基础配置文件

```bash
# 创建新的配置文件
cat << 'EOF' > ~/.zshrc
# 基础设置
export ZSH="$HOME/.oh-my-zsh"
export LANG=en_US.UTF-8
export EDITOR='nvim'
export VISUAL='nvim'

# zinit配置
source "$HOME/.local/share/zinit/zinit.git/zinit.zsh"
autoload -Uz _zinit
(( ${+_comps} )) && _comps[zinit]=_zinit

# 加载核心插件
zinit ice depth=1; zinit light romkatv/powerlevel10k  # 主题
zinit light zsh-users/zsh-autosuggestions           # 命令建议
zinit light zsh-users/zsh-syntax-highlighting       # 语法高亮
zinit light zsh-users/zsh-completions              # 补全增强
zinit light agkozak/zsh-z                          # 目录跳转

# 历史记录设置
HISTFILE="$HOME/.zsh_history"
HISTSIZE=50000
SAVEHIST=50000
setopt EXTENDED_HISTORY          # 记录命令时间戳
setopt HIST_EXPIRE_DUPS_FIRST   # 优先删除重复命令
setopt HIST_IGNORE_DUPS         # 忽略连续重复命令
setopt HIST_IGNORE_SPACE        # 忽略以空格开头的命令
setopt HIST_VERIFY              # 执行历史命令前展示
setopt INC_APPEND_HISTORY       # 实时添加历史记录
setopt SHARE_HISTORY           # 共享历史记录

# 目录设置
setopt AUTO_CD              
setopt AUTO_PUSHD          
setopt PUSHD_IGNORE_DUPS   
setopt PUSHD_MINUS         
DIRSTACKSIZE=20

EOF
```

### 4. 实用别名设置

```bash
# 添加到~/.zshrc
cat << 'EOF' >> ~/.zshrc
# 基础命令增强
alias ls='ls --color=auto'
alias ll='ls -lah'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
alias mkdir='mkdir -p'
alias df='df -h'
alias free='free -m'
alias duf='du -sh *'
alias ps='ps auxf'
alias ping='ping -c 5'
alias root='sudo -i'
alias reboot='sudo reboot'
alias poweroff='sudo poweroff'

# Git快捷命令
alias gs='git status'
alias ga='git add'
alias gaa='git add --all'
alias gc='git commit -m'
alias gp='git push'
alias gl='git pull'
alias gd='git diff'
alias gco='git checkout'
alias gb='git branch'
alias gm='git merge'
alias glog='git log --oneline --decorate --graph'

# Docker快捷命令
alias dk='docker'
alias dkc='docker-compose'
alias dkps='docker ps'
alias dkst='docker stats'
alias dktop='docker top'
alias dkimg='docker images'
alias dkpull='docker pull'
alias dkex='docker exec -it'

# 快速编辑
alias zshconfig="$EDITOR ~/.zshrc"
alias zshreload="source ~/.zshrc"
alias vimconfig="$EDITOR ~/.vimrc"
alias tmuxconfig="$EDITOR ~/.tmux.conf"
EOF
```

### 5. 实用函数

```bash
# 添加到~/.zshrc
cat << 'EOF' >> ~/.zshrc
# 创建并进入目录
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# 提取压缩文件
extract() {
    if [ -f $1 ]; then
        case $1 in
            *.tar.bz2)   tar xjf $1     ;;
            *.tar.gz)    tar xzf $1     ;;
            *.bz2)       bunzip2 $1     ;;
            *.rar)       unrar e $1     ;;
            *.gz)        gunzip $1      ;;
            *.tar)       tar xf $1      ;;
            *.tbz2)      tar xjf $1     ;;
            *.tgz)       tar xzf $1     ;;
            *.zip)       unzip $1       ;;
            *.Z)         uncompress $1  ;;
            *.7z)        7z x $1        ;;
            *)          echo "'$1' cannot be extracted" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# 快速查找文件
ff() { find . -type f -iname "*$1*" ; }
fd() { find . -type d -iname "*$1*" ; }

# 快速查看进程
psg() { ps aux | grep -v grep | grep -i -e VSZ -e "$1"; }

# 网络工具
myip() {
    curl -s http://ipecho.net/plain
    echo
}

# 快速HTTP服务器
serve() {
    local port="${1:-8000}"
    python3 -m http.server "$port"
}

# Git日志美化
gll() {
    git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit
}
EOF
```