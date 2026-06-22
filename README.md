# Hexo + GitHub Pages 个人博客

这个仓库已经初始化为一个可长期维护的 Hexo 博客项目，适合继续写 Markdown 文章并通过 GitHub Pages 发布。

## 先做的两件事

1. 修改站点配置文件 [/_config.yml](/f:/python_projects/git_blog/_config.yml)
2. 把 `url` 改成你的真实地址，例如 `https://yourname.github.io`

建议一起改掉这些字段：

- `title`
- `subtitle`
- `description`
- `author`
- `url`

## 本地常用命令

如果你的系统已经安装了 Node.js，可以直接使用：

```powershell
npm install
npx hexo new "我的第一篇文章"
npx hexo server
```

如果暂时还没全局安装 Node.js，这个项目内也带了一份本地 Node，可用下面命令：

```powershell
$env:Path = (Resolve-Path '.\.tools\node-v24.15.0-win-x64').Path + ';' + $env:Path
npm install
npx hexo new "我的第一篇文章"
npx hexo server
```

本地预览地址通常是：

```text
http://localhost:4000
```

## 写文章

新建文章：

```powershell
npx hexo new "文章标题"
```

文章会生成在：

- [/source/_posts](/f:/python_projects/git_blog/source/_posts)

因为已经开启了 `post_asset_folder: true`，每篇文章可以带自己的图片目录，更适合长期写作。

## GitHub Pages 发布

推荐使用 GitHub Actions 自动发布。

你只需要：

1. 在 GitHub 上创建仓库 `yourname.github.io`
2. 把本地项目推送到 `main` 分支
3. 进入 GitHub 仓库 `Settings -> Pages`
4. `Build and deployment` 里选择 `GitHub Actions`

工作流文件已经放在：

- [/.github/workflows/pages.yml](/f:/python_projects/git_blog/.github/workflows/pages.yml)

## 主题

当前使用 Hexo 默认主题 `landscape`。

如果后面想换主题，我们可以再一起挑一个更适合中文个人博客、维护更轻松的主题。

