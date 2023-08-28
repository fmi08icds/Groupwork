# Group 4 - Clustering

**Table of contents:**

1. [Getting started](#getting-started)
2. [Workflow](#workflow)
3. [Project goal](#project-goal)
4. [Results](#results)
5. [Members](#members)


## Getting started

### Conda

This project uses a local conda environment. It isolates the Python runtime and the required packages from the rest of the system where the code is run. To setup your local development environment, consider the following commands:

``` bash
# Create a conda environment based on the `environment.yml`. It will be stored in a local folder called `env`.
conda env create --file environment.yml --prefix ./env
## COMMENTS: this results in a ResolvePackageNotFound: for scikit-learn-intelex...on MAC the best would have been to write a bash script.. check Group3/src or provide information of the computer on which your codes were run.

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

### Repository structure

``` plain
├── 📁 clustering         <-- Package that contains implementations of clustering algorithms
│   ├── __init__.py
│   ├── dbscan.py
│   ├── utils.py
│   └── ...
│
├── 📁 env                <-- Local conda environment (not part of version control)
│   └── ...
│
├── 📁 evaluation         <-- Evaluation of clustering algorithms with different metrics
│   └── ...
│
├── 📁 notebooks          <-- Directory for Jupyter notebook files
│   └── ...
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
├── 📃 main.py            <-- Compare algorithms
│
├── 📃 pyproject.toml     <-- Configuration file
│
└── 📃 README.md          <-- Project documentation
```

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

## Project goal

**Dataset:**

- [Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) from Kaggle
- has 232,725 tracks from 26 different genres (~8,000 tracks per genre)
- each sample has 18 attributes including names (e. g. track, artists, genre), confidence measures (e. g. acousticness, instrumentalness), perceptual measures (e. g. energy, loudness) and other descriptors (e. g. tempo, duration)

**Goal:**

Find clusters of tracks with similar audio features based on the available attributes. Our hypothesis is that songs that have the same genre should also be assigned to the same cluster by the algorithm. The accuracy will be evaluated by comparing the clustering labels with the ground truth of the genre attribute.

## Results

We used the adjusted rand index (ARI) to evaluate our algorithms' performance on the dataset. Our experiments which can be reproduced by running the `main.py` script yield the following results:

|      Score | Algorithm            | Parameters                                                                            |   # Clusters | Runtime    |
|-----------:|:---------------------|:--------------------------------------------------------------------------------------|-------------:|:-----------|
| 0.0991538  | :trophy: k-Means     | `n_clusters = 25`                                                                     |           25 | 131.2 ms   |
| 0.0650288  | BIRCH                | `branching_factor = 50`<br>`threshold = 0.25`<br>`n_cluster = 25`<br>`predict = True` |           25 | 234.0 ms   |
| 0.0475148  | Affinity Propagation | `damping = 0.7`<br>`convergence_iter = 20`<br>`max_iter = 200`                        |           88 | 14252.2 ms |
| 0.00756683 | DBSCAN               | `epsilon = 0.3244`<br>`min_points = 20`                                               |            3 | 873.9 ms   |

As can be seen, the k-Means clustering algorithms produces the best results in terms of ARI metric. It also has the fastest runtime. When using the Affinity Propagation and DBSCAN algorithms the number of clusters can not be predefined. Consequently, the number of clusters found by these algorithms are largely different from the target of 25 that corresponds with the number of genres in the ground truth. In general, the results are not good, i. e., clustering is not capable of correctly classifying the songs into genres in the given dataset. Our initial assumption, that the features are similar in the same genre and different in other genres, does not hold. We presume that the classification problem can be addressed better using supervised machine learning techniques, for example Neural Networks or Random Forests.

For more details, the script `evaluation/evaluate_clusterings.py` computes more extrinsic and intrinsic clustering metrics. The numeric results are summarized in the following table.

| algorithm                           |   run_time |   rand_index_score |   adj_rand_index_score |   mut_info_score |   adj_mut_info_score |   norm_mut_info_score |   homogeneity_score |   completeness_score |   v_measure |   fowlkes_mallows_score |   silhouette_score |   calinski_harabasz_score |   davies_bouldin_score |
|:------------------------------------|-----------:|-------------------:|-----------------------:|-----------------:|---------------------:|----------------------:|--------------------:|---------------------:|------------:|------------------------:|-------------------:|--------------------------:|-----------------------:|
| K-Means                             |  0.0184274 |           0.924967 |             0.0957199  |         0.829024 |             0.247199 |             0.261424  |           0.257551  |             0.265415 |   0.261424  |                0.135115 |          0.140786  |                   707.407 |                1.68684 |
| MiniBatch K-Means                   |  0.103383  |           0.925506 |             0.114948   |         0.822396 |             0.245511 |             0.259789  |           0.255492  |             0.264234 |   0.259789  |                0.154242 |          0.14781   |                   673.727 |                1.80126 |
| Bisecting K-Means                   |  0.0888502 |           0.924711 |             0.110527   |         0.791069 |             0.236146 |             0.250733  |           0.245759  |             0.255911 |   0.250733  |                0.15026  |          0.103208  |                   596.378 |                1.96981 |
| DBSCAN (Scikit Learn)               |  0.0278918 |           0.340761 |             0.00793312 |         0.174065 |             0.087258 |             0.0908868 |           0.0540762 |             0.284659 |   0.0908868 |                0.180411 |          0.0977827 |                   245.678 |                2.39122 |
| DBSCAN (Group 4)                    |  1.20215   |           0.340761 |             0.00793312 |         0.174065 |             0.087258 |             0.0908868 |           0.0540762 |             0.284659 |   0.0908868 |                0.180411 |          0.0977827 |                   245.678 |                2.39122 |
| Affinity Propagation (Scikit-Learn) | 30.7863    |           0.955122 |             0.0416476  |         1.23054  |             0.220932 |             0.301585  |           0.382288  |             0.249016 |   0.301585  |                0.073386 |          0.100288  |                   209.937 |                1.68665 |
| Affinity Propagation (Group 4)      | 56.4165    |           0.955122 |             0.0416476  |         1.23054  |             0.220932 |             0.301585  |           0.382288  |             0.249016 |   0.301585  |                0.073386 |          0.100288  |                   209.937 |                1.68665 |
| BIRCH                               |  0.265054  |           0.895133 |             0.080145   |         0.740388 |             0.230461 |             0.246057  |           0.230014  |             0.264505 |   0.246057  |                0.137293 |          0.0931095 |                   521.584 |                1.80029 |
| BIRCH (Group 4)                     |  0.268559  |           0.904071 |             0.0996417  |         0.798106 |             0.2478   |             0.26304   |           0.247946  |             0.280093 |   0.26304   |                0.152527 |          0.11492   |                   560.341 |                1.9178  |


## Members

| Florian Winkler<br>(Group Leader) | Ralf König | Sebastian Schmidt | Clara Leidhold | Marcel Lehmann |
|:---------------------------------:|:----------:|:-----------------:|:--------------:|:--------------:|
| <img width="96" src="https://github.com/Fju.png?size=96"> | <img width="96" src="https://github.com/ralf-koenig.png?size=96"> | <img width="96" src="https://github.com/schmiseb.png?size=96"> | <img width="96" src="https://github.com/claraldh.png?size=96"> | <img width="96" src="https://github.com/Lehmsen.png?size=96"> |
