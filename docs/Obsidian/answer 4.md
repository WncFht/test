---
title: answer 4
date: 2024-11-16T20:34:02+0800
modify: 2024-12-06T00:12:56+0800
categories: graph
dir: Obsidian
share: true
tags:
  - graph
---

```thinking
你提出了很好的问题：
1. Git确实不能直接使用Obsidian的双链语法
2. 需要简化文档结构，使其更实用
3. 需要一套标准化的prompt模板

让我重新设计这些部分，使其更加实用和高效。
```

## 一、Git提交信息规范（更新版）

```bash
# 命名规范
type(scope): description

# Commit body 应包含:
- Prompt File: PROJ-123/prompts/auth-system.md
- Implementation: PROJ-123/impl/auth-core.md
- Changes: 具体改动列表

# Example:
ai-raw(auth): implement basic JWT authentication

Prompt File: AUTH-001/prompts/jwt-setup.md
Implementation: AUTH-001/impl/jwt-core.md

Changes:
- Add JWT service
- Implement token generation
- Add authentication middleware
```

## 二、Templater模板

### 1. 基础Prompt模板 (`_templates/prompt.md`)

```markdown
<%*
let ticketId = await tp.system.prompt("Ticket ID?");
let feature = await tp.system.prompt("Feature name?");
let type = await tp.system.prompt("Type? (raw/refactor/fix/test)");
_%>
---
ticket: <% ticketId %>
feature: <% feature %>
type: <% type %>
created: <% tp.date.now("YYYY-MM-DD HH:mm") %>
status: draft
---

# <% feature %> Development Prompt

## Context
- Ticket: <%= ticketId %>
- Type: <%= type %>
- Feature: <%= feature %>

## Requirements
- 
- 
- 

## Current Implementation
```typescript
// 当前代码片段（如果是修改现有代码）


## Prompt
```prompt
### System Context
You are an expert software developer. Focus on [specific domain].

### Requirements
[Key requirements here]

### Current Code
[Current code if applicable]

### Task
[Specific task description]


## Notes
- [ ] Check security implications
- [ ] Verify error handling
- [ ] Test coverage

## References
- Docs: `docs/path/to/docs`
- Code: `src/path/to/code`
```

## 三、Prompt模板库（按Git提交类型）

### 1. 新功能开发 (ai-raw)

```prompt
### System Context
You are an expert software developer working on [project/feature].
Focus on clean code, maintainability, and best practices.

### Requirements
1. Feature scope:
   [Describe feature requirements]
2. Technical constraints:
   - Language: [TypeScript/Python/etc]
   - Framework: [Framework details]
   - Standards: [Coding standards]

### Expected Deliverables
1. Core implementation
2. Type definitions
3. Error handling
4. Basic tests

### Task
Please implement [feature description] with the following considerations:
- Follow SOLID principles
- Include error handling
- Add JSDoc comments
- Include type definitions
- Implement unit tests

### Output Format
Please provide:
1. Complete implementation
2. Test cases
3. Usage examples
```

### 2. 代码重构 (ai-refactor)

```prompt
### System Context
You are a software architect specializing in code optimization and refactoring.

### Current Implementation
[Paste current code]

### Issues to Address
1. [Specific issue]
2. [Performance concern]
3. [Technical debt]

### Requirements
Please refactor the code with focus on:
1. Performance optimization
2. Code organization
3. Design patterns
4. Error handling

### Constraints
- Maintain backward compatibility
- No breaking changes
- Must pass existing tests
```

### 3. 错误修复 (ai-fix)

```prompt
### System Context
You are debugging and fixing issues in [feature/component].

### Bug Description
[Describe the bug]

### Current Behavior
[Current problematic behavior]

### Expected Behavior
[Expected correct behavior]

### Current Code
[Paste relevant code]

### Task
Please provide:
1. Root cause analysis
2. Fixed implementation
3. Prevention measures
4. Test cases
```

### 4. 测试增强 (ai-test)

```prompt
### System Context
You are a QA engineer focusing on test coverage and quality.

### Component to Test
[Component description]

### Current Implementation
[Paste code to test]

### Test Requirements
1. Unit tests
2. Edge cases
3. Error scenarios
4. Integration tests

### Expected Coverage
- [ ] Happy path
- [ ] Error handling
- [ ] Edge cases
- [ ] Performance scenarios
```

## 四、实际使用流程示例

1. **创建新功能**

```bash
# 1. 使用Templater创建文档
# 使用prompt.md模板创建新文档

# 2. 填写Prompt并获取AI回答

# 3. Git提交
git commit -m "ai-raw(auth): implement JWT authentication

Prompt File: AUTH-123/prompts/jwt-auth.md
Implementation: AUTH-123/impl/jwt-service.md

Changes:
- Add JWT service implementation
- Add token management
- Implement authentication middleware"
```

2. **重构代码**

```bash
# 1. 创建重构Prompt文档

# 2. Git提交
git commit -m "ai-refactor(auth): optimize JWT token handling

Prompt File: AUTH-123/prompts/jwt-refactor.md
Implementation: AUTH-123/impl/jwt-optimization.md

Changes:
- Improve token validation performance
- Add token caching
- Optimize error handling"
```

## 五、简化版文档结构

```
ProjectName/
├── .obsidian/
├── tickets/
│   └── PROJ-123/
│       ├── prompts/
│       │   ├── initial-setup.md
│       │   └── refactor-01.md
│       ├── impl/
│       │   ├── core.md
│       │   └── optimization.md
│       └── notes.md
├── knowledge/
│   ├── prompts/
│   │   ├── raw/
│   │   ├── refactor/
│   │   └── fix/
│   └── patterns/
└── _templates/
```

这样的结构更清晰，每个ticket一个文件夹，包含所有相关文档。

需要我详细展开任何部分吗？这个设计的主要优点是：

1. 文档结构更简单直观
2. Git提交信息更规范实用
3. Prompt模板更有针对性
4. 工作流程更加清晰