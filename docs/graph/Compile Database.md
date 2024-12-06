---
categories: graph
date: 2024-10-12T13:01:56+0800
dir: graph
modify: 2024-12-06T00:13:04+0800
share: true
tags:
  - graph
title: Compile Database
---

### 什么是 Compile Database

Compile Database 是一个 JSON 格式的文件，包含了项目中每个编译单元的编译命令和参数。Compile Database 使得你的编辑器能够获取编译信息，启用代码跳转和高亮功能，从而提高开发效率。

### 生成 Compile Database

#### 使用 CMake 生成

在 CMake 命令中添加参数：

`cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .`

#### 使用其他工具生成（如 Bear）

Bear 是一个工具，可以在构建时捕获编译命令并生成 compile_commands.json 文件。

使用方法：

`bear -- make`

## 在 VSCode 中使用 Compile Database

在cpp plugin中配置compile commands选项

`"compileCommands": "${workspaceFolder}/build/compile_commands.json"`