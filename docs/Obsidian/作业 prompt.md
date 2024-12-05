---
title: 作业 prompt
date: 2024-11-16T20:34:02+0800
modify: 2024-12-06T00:14:28+0800
categories: graph
dir: Obsidian
share: true
tags:
  - graph
---

请你用 LaTeX 输出，基于 elegantbook 模板进行排版。请遵循以下规范：

一、基础数学公式规范

1. **行内公式**：使用 `$...$` 表示行内数学公式
2. **行间公式**：
   - 基本公式：使用 `\begin{equation} ... \end{equation}`
   - 无编号公式：使用 `\begin{equation*} ... \end{equation*}`
   - 多行公式：使用 `\begin{align} ... \end{align}` 或 `\begin{split} ... \end{split}`
   - 分支公式：使用 `\begin{cases} ... \end{cases}`
3. **矩阵表示**：
   - 基本矩阵：`\begin{matrix} ... \end{matrix}`
   - 带括号：`\begin{pmatrix} ... \end{pmatrix}`
   - 带方括号：`\begin{bmatrix} ... \end{bmatrix}`

二、文档结构规范

1. **章节层级**：
   - 章：`\chapter{标题}`
   - 节：`\section{标题}`
   - 小节：`\subsection{标题}`
   - 子小节：`\subsubsection{标题}`
2. **列表环境**：
   - 有序列表：`\begin{enumerate}[label=\arabic*.] ... \end{enumerate}`
   - 无序列表：`\begin{itemize} ... \end{itemize}`
   - 描述列表：`\begin{description} ... \end{description}`

三、特殊环境规范

1. **定理类环境**：
   - 定理：`\begin{theorem} ... \end{theorem}`
   - 引理：`\begin{lemma} ... \end{lemma}`
   - 推论：`\begin{corollary} ... \end{corollary}`
   - 定义：`\begin{definition} ... \end{definition}`
2. **练习与解答**：
   - 练习：`\begin{exercise} ... \end{exercise}`
   - 解答：`\begin{solution} ... \end{solution}`
   - 示例：`\begin{example} ... \end{example}`

四、注意事项

1. 所有代码都应该放在代码块中
2. 保持一致的缩进和格式
3. 适当使用注释说明复杂的代码部分
4. 重要环境和命令要加上适当的标签供交叉引用