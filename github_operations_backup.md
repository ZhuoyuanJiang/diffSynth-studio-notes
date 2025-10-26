# GitHub Operations & Backup Strategy

## Overview

This document explains the setup and workflow for managing a local fork of DiffSynth-Studio with a private GitHub backup repository. The strategy allows you to:

- Keep your personal notes and experimental code **private**
- Maintain the ability to **sync latest changes** from the original upstream repository
- Have a **backup of your work** in a private GitHub repository
- Avoid conflicts with the original codebase (all changes are new files, not modifications)

## Architecture: Two Remote Repositories

Your local repository is connected to **two remote repositories**:

```
Local Repository: ~/projects/DiffSynth-Studio/
├─ origin      → https://github.com/zhuoyuan/diffsynth-studio-notes.git  (YOUR PRIVATE REPO)
└─ upstream    → https://github.com/modelscope/DiffSynth-Studio.git      (ORIGINAL REPO - READ ONLY)
```

### Understanding Remotes

A **remote** is simply a reference to a GitHub repository. When you `git push`, you push to `origin` by default. When you `git fetch`, you can fetch from either `origin` or `upstream`.

| Remote | Purpose | Permissions | Usage |
|--------|---------|-------------|-------|
| **origin** | Your private backup repo | Read + Write | Store your notes, experimental code, and local changes |
| **upstream** | Original DiffSynth-Studio repo | Read-only | Sync latest updates from the main project |

## Initial Setup (One-Time)

### Step 1: Create a Private GitHub Repository

1. Log in to [GitHub](https://github.com)
2. Click **+** (top right) → **New repository**
3. Fill in:
   - **Repository name**: `diffsynth-studio-notes`
   - **Description**: `Local notes and experimental code for DiffSynth-Studio`
   - **Private** ← Select this option!
   - Keep other settings as default
   - Click **Create repository**

### Step 2: Configure Local Repository Remotes

```bash
cd ~/projects/DiffSynth-Studio

# Verify current remotes (should only show origin pointing to modelscope)
git remote -v

# Rename the original 'origin' to 'upstream'
git remote rename origin upstream

# Add your private repository as the new 'origin'
git remote add origin https://github.com/zhuoyuan/diffsynth-studio-notes.git
# (Replace 'zhuoyuan' with your GitHub username)

# Verify the new setup
git remote -v
```

**Expected output:**
```
origin      https://github.com/zhuoyuan/diffsynth-studio-notes.git (fetch)
origin      https://github.com/zhuoyuan/diffsynth-studio-notes.git (push)
upstream    https://github.com/modelscope/DiffSynth-Studio.git (fetch)
upstream    https://github.com/modelscope/DiffSynth-Studio.git (push)
```

### Step 3: Create a Dedicated Branch for Your Work

```bash
# Create and switch to a new branch for your notes and changes
git checkout -b notes-and-scripts

# Add your notes and local files
git add *.md
git add examples/wanvideo/model_training/commented_*.py
git add diffsynth/pipelines/commented_*.py
git add diffsynth/trainers/commented_*.py
git add diffsynth/utils/commented_*.py
# Add any other local files you want to track

# Commit your changes
git commit -m "Add private notes and experimental scripts"

# Push to your private repository (first time use -u flag)
git push -u origin notes-and-scripts
```

## Daily Workflow

### Scenario A: Update Your Notes and Experimental Code

```bash
# Make sure you're on the notes-and-scripts branch
git checkout notes-and-scripts

# Make your edits to notes, training configs, or experimental scripts
# ...

# Stage and commit your changes
git add .
git commit -m "Update training notes - added analysis of loss curves"

# Push to YOUR private repository (origin)
# This is where all your personal work is backed up
git push origin notes-and-scripts
```

**Remember**: `git push` automatically pushes to `origin` (your private repo). Your work is safe and private!

### Scenario B: Sync Latest Changes from Original Repository

The original DiffSynth-Studio repository gets updates. You want to pull the latest code while keeping your notes intact.

```bash
# Step 1: Fetch the latest code from the original repository
git fetch upstream
# This downloads all updates from upstream without modifying your local branches

# Step 2: Switch to main branch
git checkout main

# Step 3: Merge the latest upstream code into your local main
git merge upstream/main
# OR use rebase for a cleaner history (slightly advanced)
# git rebase upstream/main

# Step 4: Update your notes-and-scripts branch to be based on the latest main
git checkout notes-and-scripts
git rebase main

# Step 5: Push your updated branch back to your private repository
git push origin notes-and-scripts
```

**What happened**:
- ✅ Your `main` branch is now up-to-date with the original repository
- ✅ Your `notes-and-scripts` branch is now based on the latest code
- ✅ Your personal notes and changes are still intact and backed up

### Scenario C: Create a New Experimental Branch

```bash
# Make sure you're up-to-date with the latest code first
git checkout main
git pull upstream main

# Create a new experimental branch
git checkout -b experiment/new-training-strategy

# Make your changes and commits
# ...
git add .
git commit -m "Test new data augmentation pipeline"

# Push to your private repository
git push -u origin experiment/new-training-strategy
```

## Complete Workflow Example

Here's a typical day of work:

```bash
# Morning: Sync with latest upstream changes
cd ~/projects/DiffSynth-Studio
git fetch upstream
git checkout main
git merge upstream/main           # Get latest code from original repo

# Update your main working branch
git checkout notes-and-scripts
git rebase main
git push origin notes-and-scripts

# During the day: Make improvements and save progress
# (Edit files...)
git add .
git commit -m "Improve RenderMe360 data loading"
git push origin notes-and-scripts

# Continue working and pushing periodically
# (Edit files again...)
git add .
git commit -m "Add validation metrics for video quality"
git push origin notes-and-scripts

# End of day: Everything is backed up in your private repository!
```

## Common Commands Reference

| Task | Command | Notes |
|------|---------|-------|
| **View all remotes** | `git remote -v` | Shows both origin and upstream |
| **Fetch latest upstream** | `git fetch upstream` | Get new code, don't modify local branches |
| **Update main branch** | `git merge upstream/main` | Merge latest upstream into main |
| **Switch to your work branch** | `git checkout notes-and-scripts` | Go back to your personal branch |
| **Save your work** | `git push origin notes-and-scripts` | Push to YOUR private repo |
| **Check branch status** | `git branch -a` | See all local and remote branches |
| **View commits** | `git log --oneline -10` | See recent commits on current branch |

## Important Notes

### ✅ Best Practices

1. **Always fetch before merging**: `git fetch upstream` before `git merge upstream/main`
2. **Use descriptive commit messages**: Makes it easy to understand what you changed
3. **Push frequently**: Don't wait to push; save your work regularly to the private repo
4. **Keep main clean**: Only merge upstream into main, don't make custom changes there
5. **Use branches for experiments**: Create new branches (`experiment/xxx`) for risky changes

### ❌ Common Mistakes to Avoid

1. **Don't accidentally push to upstream**: You can't anyway (no permissions), but just be aware
2. **Don't lose the upstream connection**: Don't rename or remove the upstream remote
3. **Don't mix origin and upstream**: Remember `origin` = private (push here), `upstream` = original (fetch from here)
4. **Don't force push on shared branches**: Avoid `git push -f` unless you know what you're doing

## Troubleshooting

### "I accidentally modified main branch"

```bash
# Don't panic! You can reset it to upstream
git checkout main
git reset --hard upstream/main
# This discards any changes you made to main
```

### "I want to see what's different between my branch and upstream"

```bash
git diff main upstream/main
git log upstream/main..main      # Commits in main not in upstream
```

### "I want to see all my commits in notes-and-scripts"

```bash
git checkout notes-and-scripts
git log --oneline -20            # Show last 20 commits
git log --oneline upstream/main..HEAD  # Show commits not in upstream
```

### "I need to go back to a previous commit"

```bash
# See the commit history
git log --oneline

# Go back to a specific commit (safe, doesn't delete history)
git reset --soft COMMIT_HASH     # Keep changes
git reset --hard COMMIT_HASH     # Discard changes
```

## Summary

Your setup uses a **dual-remote strategy**:

```
┌─────────────────────────────────────────────┐
│ Your Local Repository                       │
│ (main branch)  ←←← (fetch/merge)            │
│ (notes-and-scripts) ←→ (push/pull)          │
└─────────────────────────────────────────────┘
        ↑                           ↓
    (read-only)                 (read-write)
        │                           │
        │                           │
    upstream                      origin
    (modelscope)            (your-private-repo)
    DiffSynth-Studio        diffsynth-studio-notes
    (ORIGINAL)              (YOUR BACKUP)
```

**Key Rule**:
- 🔽 **Pull/Fetch from upstream** (original repository)
- 🔼 **Push to origin** (your private repository)
- 📝 **Your notes and changes stay private and safe**

Happy coding! 🚀
