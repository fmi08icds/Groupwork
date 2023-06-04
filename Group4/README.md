# Group 4 - Clustering

## Getting started

### Conda

This project uses a local conda environment. It isolates the Python runtime and the required packages from the rest of the system where the code is run. To setup your local development environment, consider the following commands:

``` bash
# Create a conda environment based on the `environment.yml`. It will be stored in a local folder called `env`.
conda env create --file environment.yml --prefix ./env

# To activate the new conda environment, use
conda activate ./env

# The shell prompt will now show the complete path to the conda environment which can be pretty long.
# Hence, it is recommended to adjust the prompt to only show the directory name. This must only be configured once.
# You may need to deactivate and activate the conda env for changes to take effect.
conda config --set env_prompt '({name}) '

# In case more packages are needed, install them using conda or pip
# Make sure to update the `environment.yml` file, so that others can update their conda env
conda install scikit-learn

# If the `environment.yml` has been changed by others, you can update your own env using
conda env update --file environment.yml --prefix ./env
```

### Tooling

The conda setup will also install some tools to increase productivity. As of now, they are `pytest` for running unit tests and `black` for automatically formatting Python code. Feel free to suggest other helpful tools.

It is recommended to run both tools before committing to ensure a consistent formatting and to avoid broken code.

``` bash
# Format all scripts in the `clustering` and `tests` folder
black clustering/ tests/
# Format a single file specifically
black clustering/dbscan.py

# Run all tests scripts in the `tests` folder
pytest
```
These tools are configured by the `pyproject.toml` file.

## Project structure

``` plain
├── 📁 clustering         <-- Package that contains implementations of clustering algorithms
│   ├── __init__.py
│   ├── dbscan.py
│   ├── utils.py
│   └── ...
│
├── 📁 env                <-- Local conda environment (not part of version control)
│
├── 📁 tests              <-- Unit tests scripts
│   ├── __init__.py
│   ├── test_dbscan.py
│   ├── test_utils.py
│   └── ...
│
├── 📃 .gitignore         <-- List of files and folders that should be ignored by git
│
├── 📃 environment.yml    <-- List of python dependencies for the conda environment
│
├── 📃 main.ipynb         <-- Project results, visualizations and documentation
│
├── 📃 pyproject.toml     <-- Configuration file
│
└── 📃 README.md          <-- Project documentation
```

## Project goals

TODO

## Workflow

:chestnut: In a nutshell:

- Create an issue for each open task, make use of the template below
- Look into the [Kanban board](https://github.com/orgs/fmi08icds/projects/4) to see which tasks are unassinged and in status *Todo*
- Grab a task that you are able to accomplish, assign it to yourself and change the status to *Work in progress*
- Create your own branch and push your commits to it
- When you are done, create a Pull Request to merge your work into the main branch of the group
- Group leader reviews your Pull Request and merges it
- Change the status of your task to *Done* if the Pull Request has been successfully merged
- Repeat until all tasks are done

### Branches and Pull Requests

The main branch of this group is called `group-4`. It is a protected branch such that no commits can be pushed to it directly. Instead a [Pull Request](https://github.com/fmi08icds/Groupwork/pulls) must be created and approved by the group leader, to merge a feature branch into the `group-4` branch. Besides that, there are no restrictions when creating feature branches. Make sure that everything is safe before creating the Pull Request.

### Issues

All work packages (tasks) are described as [Issues](https://github.com/fmi08icds/Groupwork/issues). When writing an Issue please consider the following:

- Make sure that the workload is not too big. As a rule of thumb, a single person should be able to complete it in less than a week
- Provide a short description of what needs to be done
- Provide a list of Acceptance Criteria. These are requirements that need to be met to call the task done
- Assign the Issue to the project "Group 4 - Clustering" and set the status to "Todo"
- Do not assign the issue to anyone unless it is already known who is responsible for the task

Here is a template with an example content:

<details>

<summary>Markdown Template for Issues</summary>

``` md
**Description:**

Currently, our dataset mostly contains pictures of cats with dark fur.
I observed that the classification accuracy suffers from this bias.
We should add more samples to the dataset that show cats with different colored fur.

**Acceptance criteria**:

- 1000 samples were added to the dataset
- It was assured that these new samples show cats with bright colored fur
- The new samples are considered when creating the train/test split and executing `train.py`
```

</details>

### Projects

Issues are organized in a [Kanban board](https://github.com/orgs/fmi08icds/projects/4). The board consists of three columns that represent the statuses *Todo*, *Work in progress* and *Done*. The status can be changed by simply dragging the cards into another column.

If you want to work on a task, look for unassigned issues in the *Todo* column. Read the description and acceptance criteria of the issue. If you feel like you are able to work on this task, assign it to yourself and change the status to *Work in progress*. In case the goal of the task is unclear, ask the creator of the issue for clarification. Once you have completed the task, congratulations, you can change the status to *Done* :+1:

## Members

| Florian Winkler<br>(Group Leader) | Ralf König | Sebastian Schmidt | Clara Leidhold | Marcel Lehmann |
|:---------------------------------:|:----------:|:-----------------:|:--------------:|:--------------:|
| <img width="96" src="https://github.com/Fju.png?size=96"> | <img width="96" src="https://github.com/ralf-koenig.png?size=96"> | <img width="96" src="https://github.com/schmiseb.png?size=96"> | <img width="96" src="https://github.com/claraldh.png?size=96"> | <img width="96" src="https://github.com/Lehmsen.png?size=96"> |
