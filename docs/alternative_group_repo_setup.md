# Alternative Private Group Repository Setup

*This document contains the legacy/alternative instructions for creating a private group repository instead of using the public "Fork & Upstream" workflow.*

## Create your group's private repository and clone it

Use **one independent private GitHub repository per group**, initialized from a clone of the public course repository. Each group member then works in a local clone of the private repository. This preserves the shared Git history needed to merge future course updates while keeping your group's work visible only to people you invite.

A GitHub fork of the public course repository is also public and cannot be made private. For private work, follow the steps below instead of clicking **Fork**. See [GitHub's fork visibility guidance](https://docs.github.com/en/pull-requests/reference/forks#visibility-of-forks).

1. Choose one group member to own the repository in their personal GitHub account. On GitHub, select **New repository** and set its visibility to **Private**.
2. In **Repository name**, use `IP-HS26-MehrSpur-CaseStudy-Lastname1-Lastname2-Lastname3`, including every member's last name in alphabetical order. Use hyphens between names and ASCII spellings, for example `IP-HS26-MehrSpur-CaseStudy-Mueller-Rossi-Smith`. For individual work, use only your own last name.
3. Create an **empty** repository: do not initialize it with a README, license, `.gitignore` or template. The course clone supplies these files and its existing history.
4. The owner runs the following commands once to populate the private repository. Replace `GROUP-OWNER` and `GROUP-REPOSITORY` with the actual GitHub username and repository name. The commands in this guide assume the course's default branch is `main`; substitute its actual name if different.

```bash
git lfs install
git clone --branch main https://github.com/InfrastructurePlanningREISETHZ/IP-HS26-MehrSpur-CaseStudy.git GROUP-REPOSITORY
cd GROUP-REPOSITORY
git remote rename origin upstream
git remote add origin https://github.com/GROUP-OWNER/GROUP-REPOSITORY.git
git lfs fetch --all upstream main
git lfs push --all origin main
git push -u origin main
git remote -v
```

The Git LFS commands copy the large files referenced by the course history into the group's own repository storage. This uses the group repository owner's LFS allowance. If an upload fails because of a storage or bandwidth limit, contact the teaching team before continuing; the data files must be available to teammates and reviewers.

After the initial push succeeds, the owner invites the other group members through **Settings → Collaborators → Add people**. Once they accept, the **other members** clone this private repository and add the course remote:

```bash
git lfs install
git clone https://github.com/GROUP-OWNER/GROUP-REPOSITORY.git
cd GROUP-REPOSITORY
git remote add upstream https://github.com/InfrastructurePlanningREISETHZ/IP-HS26-MehrSpur-CaseStudy.git
git lfs pull
git remote -v
```

Check that `origin` points to your **private group repository** and `upstream` points to the **official public course repository**. Push your group's work to `origin`. Fetching or merging from `upstream` does not publish your work or give the course owners access to your private repository. See [GitHub's remote repository guide](https://docs.github.com/en/get-started/git-basics/managing-remote-repositories).

Git LFS downloads the large data files. If you already have a clone, run `git lfs install` and `git lfs pull` inside it to retrieve those files.

Before updating, save and commit your work or stash unfinished changes. If a command reports a merge conflict, stop and resolve it before continuing; ask the teaching assistants for help if needed.

At the beginning of each phase, **one group member** incorporates the latest course changes into the private group repository:

```bash
git switch main
git pull --no-rebase origin main
git fetch upstream
git lfs fetch --all upstream upstream/main
git merge upstream/main
git lfs checkout
git lfs push --all origin main
git push origin main
```

After that member has pushed the update, the other members update their local copies:

```bash
git switch main
git pull --no-rebase origin main
git lfs pull
```

If you stashed unfinished work, restore it with `git stash pop` after updating and resolve any conflicts. Coordinate edits to the same notebook to reduce merge conflicts. For parallel work, use branches and pull requests within the private group repository.
