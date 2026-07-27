# 重构前备份记录

创建日期：2026-07-27

## 原始状态

- 原始分支：`main`
- 原始 commit：`7e0020ae4267961ca95baaeaab953f02a08f46db`
- snapshot commit：`7e0020ae4267961ca95baaeaab953f02a08f46db`
- 说明：重构前没有未提交的源码、测试、脚本或文档改动，因此 snapshot 与原始 commit 相同，没有创建无内容提交。

## 三层备份

- 备份分支：`backup/pre-refactor-20260727`
- 备份分支 SHA：`7e0020ae4267961ca95baaeaab953f02a08f46db`
- 注释标签：`pre-refactor-20260727`
- 标签指向 commit：`7e0020ae4267961ca95baaeaab953f02a08f46db`
- Git bundle：`C:\Users\苒苒\Documents\TorF_reproduction_pre_refactor_20260727.bundle`
- 未跟踪资产：`C:\Users\苒苒\Documents\TorF_reproduction_pre_refactor_assets_20260727`
- 原始未跟踪文件清单：`PRE_REFACTOR_UNTRACKED_FILES.txt`

资产备份共 75 个文件、32,299,570 字节。普通日志、ZIP、pstats、pyc 和缓存不在资产备份范围内。两个 checkpoint 已逐个比较 SHA256，备份与原件一致。

## 验证结果

- `git show backup/pre-refactor-20260727`：成功。
- `git show pre-refactor-20260727`：成功。
- `git bundle verify ..\TorF_reproduction_pre_refactor_20260727.bundle`：成功，bundle 包含完整历史。
- checkpoint SHA256：
  - `diagnostic_500_20260727_205411/checkpoint.pkl`：`1105B255AE51793E2A86D556AB9B7E46D25125CB23F13CCD3B5E926DACBF3C22`
  - `resume_500_to_550_20260727_215030/checkpoint.pkl`：`DF55386491757B0EC1402BD9C897E7B264B98EF08ED4A455460E884E3FFD07FD`

## 恢复命令

从本仓库恢复原始代码：

```powershell
git switch main
git reset --hard backup/pre-refactor-20260727
```

为避免覆盖现有工作，也可以创建独立恢复分支：

```powershell
git switch -c recovery/pre-refactor backup/pre-refactor-20260727
```

从 bundle 创建全新仓库：

```powershell
git clone C:\Users\苒苒\Documents\TorF_reproduction_pre_refactor_20260727.bundle TorF_reproduction_recovered
Set-Location TorF_reproduction_recovered
git switch main
```

恢复未跟踪资产时，从以下目录按需复制，不要覆盖较新的运行结果：

```text
C:\Users\苒苒\Documents\TorF_reproduction_pre_refactor_assets_20260727
```
