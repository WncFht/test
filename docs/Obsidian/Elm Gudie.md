---
title: Elm Gudie
date: 2024-09-15T14:51:53+0800
modify: 2024-12-06T00:10:27+0800
categories: Computer
dir: Obsidian
share: true
tags:
  - Computer
  - Advanced
  - 函数式编程
---

# Elm Gudie

## 1 An Introduction to Elm

- No runtime errors in practice
- Friendly error messages
- Reliable refactoring
- Automatically enforced semantic cersioning for all Elm packages

## 2 Core Language

#### 2.1.1 Values

using (++) operator to put some strings together

#### 2.1.2 Functions

 using fewer parentheses and commas

 ```
madlib "cat" "ergonomic"                  -- Elm
madlib("cat", "ergonomic")                // JavaScript

madlib ("butter" ++ "fly") "metallic"      -- Elm
madlib("butter" + "fly", "metallic")       // JavaScript
```

#### 2.1.3 If Expressions

#### 2.1.4 Lists

#### 2.1.5 Tuples

#### 2.1.6 Records

the same as Dictionaries

## 3 The Elm Architecture

a pattern for architecting interactive programs

#### 3.1.1 The Basic Pattern

3 parts:

- Model
	- the state of your application
- View
	- a way to turn your state into HTML
- Update
	- a way to update your state based on messages  
These three concepts are the core of The Elm Architecture.

### 3.2 Buttons

### 3.3 Text Fields

### 3.4 Forms

## 4 Types

### 4.1 Reading Types

### 4.2 Type Aliases

### 4.3 Custom Types

### 4.4 Pattern Matching

## 5 Error Handing

### 5.1 Maybe

### 5.2 Result

## 6 Commands and Subscriptions

### 6.1 HTTP

### 6.2 JSON

### 6.3 Random

### 6.4 Time

## 7 Installation

### 7.1 Code Editor

### 7.2 Elm

need proxy

```
set HTTP_PROXY=http://127.8.8.1:10809
```

## 8 JavaScript Interop

### 8.1 Flags

### 8.2 Ports

### 8.3 Custom Elements

### 8.4 Limits

## 9 Web Apps

### 9.1 Navigation

### 9.2 URL Parsing

### 9.3 Modules

### 9.4 Structure

## 10 Optimization

### 10.1 Html. lazy

### 10.2 Html. keyed

### 10.3 Minification

## 11 Next Steps

## 12 Appendix

### 12.1 Types as Sets

### 12.2 Types as Bits

### 12.3 Function Types