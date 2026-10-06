# Git cheat sheet for our group

## Remotes

- `origin`: our group repo (`omaghur/IP-Mobility-Hubs`). We pull from it and push to it.
- `upstream`: the course repo. We only pull from it to get course updates.

Check them with `git remote -v`.

## One-time setup

```bash
git lfs install
git clone https://<your-github-username>@github.com/omaghur/IP-Mobility-Hubs.git
cd IP-Mobility-Hubs
git remote add upstream https://github.com/InfrastructurePlanningREISETHZ/IP-HS26-MehrSpur-CaseStudy.git
git lfs pull
git config user.name "<your-github-username>"
git config user.email "<email on your GitHub account>"
```

When Git asks for a password, paste a GitHub **Personal Access Token** (with `repo` scope), not your account password.

## Start a task

```bash
git switch main
git pull --no-rebase origin main
git switch -c <your-name>/<task>
```

## Save and publish work

```bash
git status
git add <files>
git commit -m "Describe the change"
git push -u origin <your-name>/<task>
```

Then open a pull request into `main` on GitHub.

## After your PR is merged

```bash
git switch main
git pull --no-rebase origin main
git branch -d <your-name>/<task>
```

## Course updates (start of each phase)

One designated member:

```bash
git switch main
git pull --no-rebase upstream main
git lfs pull
git push origin main
```

Everyone else:

```bash
git switch main
git pull --no-rebase origin main
git lfs pull
```

## Submit

```bash
git switch main
git pull --no-rebase origin main
git push origin main
git rev-parse HEAD
```

Submit the repo URL, the full 40-character SHA and all members' names. Before submitting, open `https://github.com/omaghur/IP-Mobility-Hubs/tree/<SHA>` and check that all deliverables are there.

## Rules

- Pull before creating a branch.
- Agree who edits which notebook. Notebook conflicts are hard to resolve.
- Check `git status` before every commit and add only the files you mean to.
- Never commit tokens, `.venv/`, caches or generated outputs.
- Never force-push `main`.
- If you get a conflict, stop and ask the group. Don't guess.

## Unstuck

```bash
git merge --abort        # cancel a merge that went wrong
git restore <file>       # discard your uncommitted changes to a file
git stash                # put uncommitted work aside
git stash pop            # bring it back
```
