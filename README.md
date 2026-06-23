# 山高路远博客说明

这个仓库是一个基于 `Hexo + GitHub Pages` 的个人博客项目。

当前已经完成：

- 本地初始化 Hexo 博客
- 配置 GitHub Pages 自动发布
- 主题切换为 `Fluid`
- 启用本地搜索
- 新增 `About`、`Categories`、`Tags` 页面
- 修复 GitHub Actions 构建时区为 `Asia/Shanghai`

站点地址：

- `https://toseethebiggerworld.github.io/`

## 仓库里几个重要文件

- [/_config.yml](/f:/python_projects/git_blog/_config.yml)
  Hexo 主配置，包含站点标题、作者、网址、语言、时区、主题。

- [/_config.fluid.yml](/f:/python_projects/git_blog/_config.fluid.yml)
  Fluid 主题配置，包含首页横幅、关于页、搜索、目录、深色模式等。

- [/.github/workflows/pages.yml](/f:/python_projects/git_blog/.github/workflows/pages.yml)
  GitHub Pages 自动发布工作流。每次 push 到 `main` 后会自动构建并发布。

- [/source/_posts](/f:/python_projects/git_blog/source/_posts)
  博客文章目录。以后写文章主要就在这里。

## 这次从零搭建的工作流程记录

1. 在本地创建 Hexo 博客项目
2. 配置主站点信息和 GitHub Pages 地址
3. 初始化 Git 仓库并绑定远程 GitHub 仓库
4. 配置 GitHub Actions 自动发布 Pages
5. 将主题从 `landscape` 更换为 `Fluid`
6. 安装 `hexo-generator-search`，支持站内搜索
7. 新建关于页、分类页、标签页
8. 修复 GitHub Actions 构建时区，避免文章日期比本地少一天

## 以后怎么写博客

### 1. 新建文章

如果系统已经安装 Node.js：

```powershell
npx hexo new "文章标题"
```

如果暂时还使用仓库内置 Node：

```powershell
$env:Path = (Resolve-Path '.\.tools\node-v24.15.0-win-x64').Path + ';' + $env:Path
.\node_modules\.bin\hexo.cmd new "文章标题"
```

文章会生成到：

- [/source/_posts](/f:/python_projects/git_blog/source/_posts)

因为已经开启 `post_asset_folder: true`，每篇文章可以带自己的资源目录，适合放图片。

### 2. 编辑文章

直接打开 `source/_posts` 里的 Markdown 文件编辑即可。

文章头部通常长这样：

```yaml
---
title: 文章标题
date: 2026-06-22 12:00:00
tags:
  - 标签1
  - 标签2
categories:
  - 分类名
---
```

正文直接写 Markdown。

### 3. 本地预览

如果系统已经安装 Node.js：

```powershell
npm install
npx hexo clean
npx hexo server
```

如果使用仓库内置 Node：

```powershell
$env:Path = (Resolve-Path '.\.tools\node-v24.15.0-win-x64').Path + ';' + $env:Path
.\node_modules\.bin\hexo.cmd clean
.\node_modules\.bin\hexo.cmd server
```

本地预览地址：

- `http://localhost:4000`

### 4. 生成静态页面

```powershell
.\node_modules\.bin\hexo.cmd generate
```

生成结果会放到：

- [/public](/f:/python_projects/git_blog/public)

## 以后怎么上传博客

你的博客现在是通过 **GitHub Actions 自动发布** 的，所以以后不需要手动上传 `public` 目录。

你只需要：

1. 写或修改文章
2. 本地预览确认没问题
3. 提交 Git
4. push 到 GitHub
5. 等待 Actions 自动发布

最常用的命令就是：

```powershell
git add .
git commit -m "更新博客"
git push
```

推送完成后：

1. 打开 GitHub 仓库的 `Actions`
2. 等待 `Deploy Hexo to Pages` 工作流运行完成
3. 再访问站点

## 如果页面看起来没更新

优先按下面顺序检查：

1. 看 GitHub 仓库 `Actions` 是否成功
2. 强制刷新浏览器
   Windows 常用：`Ctrl + F5` 或 `Ctrl + Shift + R`
3. 确认访问的是正式地址：
   `https://toseethebiggerworld.github.io/`

## 常见维护操作

### 修改博客标题

编辑：

- [/_config.yml](/f:/python_projects/git_blog/_config.yml)

修改：

- `title`
- `author`
- `description`

### 修改主题首页文案

编辑：

- [/_config.fluid.yml](/f:/python_projects/git_blog/_config.fluid.yml)

可以重点改这些字段：

- `navbar.blog_title`
- `index.slogan.text`
- `about.name`
- `about.intro`

### 修改关于页内容

编辑：

- [/source/about/index.md](/f:/python_projects/git_blog/source/about/index.md)

## 一套最短发布流程

以后每次发博客，按这套最短流程走就可以：

1. `git pull`
2. 新建或修改文章
3. 本地预览
4. `git add .`
5. `git commit -m "更新博客"`
6. `git push`
7. 等待 GitHub Actions 发布完成

