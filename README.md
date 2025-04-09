# SARI Data Pipeline

A data preprocessing pipeline for SARI data.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sari-data-pipeline.git
cd sari-data-pipeline
```

2. Install uv if you haven't already:
```bash
pip install uv
```


3. Create and activate a virtual environment using `uv`:
```bash

uv venv --python 3.12 # or any version above 3.8
source .venv/bin/activate
```

5. Install the required dependencies using uv:
```bash
uv pip install -r requirements.txt
```

The pipeline should now be ready to use. You can verify the installation by running:
```bash
python main.py
```

## Configuration System

The pipeline uses a flexible configuration system that allows you to:

1. Define default parameters in a configuration file (YAML or JSON)
2. Override parameters via command-line arguments




### Command-Line Usage

You can run the pipeline with the default configuration:

```bash
python main.py
```

Or specify a custom configuration file:

```bash
python main.py --config my_config.yaml
```

You can also override specific parameters:

```bash
python main.py --data-path custom_data.jsonl --output-path custom_output.jsonl --deduplicate-threshold 0.9
```


## Contribution to the Repository
- **DO NOT push TO main / dev branch directly**
- Always PR to **dev** branch first.

### 1. Create a New Branch for Your Work

Always create a new branch from the latest `dev` branch before making any changes.

```bash
# Make sure you are in the dev branch
git checkout dev

# Ensure it's up to date （see below for more details)
git pull

# Create a new branch for your feature or fix
git checkout -b feature-branch-name
```

#### `git pull` vs. `git pull origin dev`
If your `main` branch is properly set to track `origin/dev`, you can simply use:
```bash
git pull
```
This will pull the latest changes from the remote repository automatically. 
However, if you want to explicitly specify the remote and branch, you can use:
```bash
git pull origin dev
```
To check if your `dev` is tracking `origin/dev`, run:
```bash
git branch -vv
```
If it shows `dev` tracking `origin/dev`, then `git pull` alone is sufficient.

### 2. Make Your Changes

Modify your files as needed. Once done, add and commit your changes.

*Note on code quality*: Ideally, you should at least check the format of your code, e.g., type hinting, redundant imports, using tools like `Ruff`,before make your commits.

```bash
ruff format your_script.py # or  . for the whole project
ruff check --fix your_script.py # or  . for the whole project
```

```bash
# Stage specific changes
git add file1 file2

# Or stage all changes
git add .

# Commit your changes with a meaningful message
git commit -m "Brief description of your changes"
```

### 3. Push the Branch to Your Repository

After committing, push your branch to GitHub.

```bash
# Push the new branch
git push origin feature-branch-name
```

### 4. Create a Pull Request (PR)

1. Go to your GitHub repository.
2. Navigate to your pushed branch.
3. Click **New Pull Request**.
4. Ensure the base repository and branch are correct.
5. Add a descriptive title and details about your changes.
6. Request a reviewer to review the PR before merging.
7. Submit the pull request.

### 5. Keep Your Branch Up-to-Date (Rebase)

If the `dev` branch has new updates, you need to rebase to avoid conflicts before your PR being merged into `dev`.

```bash
# Switch to main and update it
git checkout dev
git pull

# Switch back to your feature branch
git checkout feature-branch-name

# Rebase your branch on top of the latest main
git rebase dev
