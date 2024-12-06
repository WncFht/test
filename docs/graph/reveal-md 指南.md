---
categories: graph
date: 2024-11-25T08:40:14+0800
dir: graph
modify: 2024-12-06T00:13:46+0800
share: true
tags:
  - graph
title: reveal-md 指南
---

# 使用 Reveal-md 制作幻灯片的完整指南

## 1. 安装 Node.js 和 npm

在开始之前，请确保已安装 [Node.js](https://nodejs.org/) 和 npm。可以通过以下命令检查是否安装：

```bash
node -v
npm -v
```

如果未安装，请到 [Node.js 官方网站](https://nodejs.org/) 下载并安装。

---

## 2. 更改 npm 源（可选，适用于国内用户）

国内用户可能会遇到 npm 安装速度较慢的问题，可以将 npm 的默认源更改为国内镜像（如淘宝源）。

### 更换为淘宝源：

```bash
npm config set registry https://registry.npmmirror.com
```

### 验证是否生效：

```bash
npm config get registry
```

输出类似如下内容即表示成功：

```
https://registry.npmmirror.com/
```

若以后需要恢复官方源，可以使用以下命令：

```bash
npm config set registry https://registry.npmjs.org
```

---

## 3. 安装 Reveal-md

使用 npm 全局安装 `reveal-md`：

```bash
npm install -g reveal-md
```

安装完成后，检查是否成功：

```bash
reveal-md --version
```

如果能输出版本号，说明安装成功。

---

## 4. 创建 Markdown 文件

使用任何文本编辑器（如 VSCode）创建一个 `.md` 文件，例如 `slides.md`，并编写以下内容：

```markdown
# 我的第一张幻灯片

---

## 第二张幻灯片

- 这是一个列表
- 第二项

---

## 使用代码块

```javascript
console.log("Hello, Reveal-md!");
``

```

---

## 5. 运行 Reveal-md

在终端中导航到 `.md` 文件所在目录，然后运行以下命令启动幻灯片：

```bash
reveal-md slides.md
```

默认情况下，幻灯片会在浏览器中打开。

---

## 6. 自定义样式和主题

### 更改主题

可以通过命令行参数更改主题。例如：

```bash
reveal-md slides.md --theme solarized
```

Reveal.js 提供了多种主题，如：

- `black`
- `white`
- `league`
- `moon`
- `night`

### 添加自定义 CSS

创建一个自定义 CSS 文件（如 `custom.css`），并在启动命令中指定：

```bash
reveal-md slides.md --css custom.css
```

---

## 7. 导出为 HTML 或 PDF

### 导出 HTML

运行以下命令，将幻灯片导出为静态 HTML 文件：

```bash
reveal-md slides.md --static
```

生成的文件会保存在 `./_static` 目录中。

### 导出 PDF

确保已安装 Puppeteer，然后运行以下命令导出 PDF：

```bash
reveal-md slides.md --print slides.pdf
```

---

## 8. 高级功能

### 分栏布局

使用 `:::` 分隔符实现分栏布局：

```markdown
## 分栏示例

::: left
这是左栏内容
:::

::: right
这是右栏内容
:::
```

### 使用自定义配置文件

创建一个 `reveal-md.json` 文件，用于配置全局选项：

```json
{
  "theme": "night",
  "highlightTheme": "dracula",
  "revealOptions": {
    "transition": "fade"
  }
}
```

启动时指定配置文件：

```bash
reveal-md slides.md --config reveal-md.json
```

---

## 9. 常见问题

### 问题 1：安装失败或网络超时

- 请检查是否更换了 npm 源。
- 尝试加上 `--legacy-peer-deps` 参数重新安装：

  ```bash
  npm install -g reveal-md --legacy-peer-deps
  ```

### 问题 2：幻灯片无法正常渲染

- 确保 Markdown 文件的格式正确，使用 `---` 分隔页面。
- 检查浏览器控制台的报错信息。

---

## 10. 卸载 Reveal-md

如果需要卸载，可以运行以下命令：

```bash
npm uninstall -g reveal-md
```

---

按照上述步骤即可快速上手 `reveal-md`，并创建美观的 Markdown 幻灯片！