# Regression Comparison

This projects aims at comparing different regression models. Every member of our
group implements a own regression model.

## Getting Started

For the best development experience please use VSCode and open this directory as the
root workspace directory.

To set up your local development environment, please use a fresh virtual environment (`python -m venv .venv`), then run:

    pip install -r requirements.txt -r requirements-dev.txt
    pip install -e .

The first command will install all requirements for the application and to execute tests. With the second command, you'll get an editable installation of the module, so that imports work properly.

You can now access the CLI with `python -m regression_comparison`.

### Testing

We use `pytest` as test framework. To execute the tests, please run

    pytest tests

To run the tests with coverage information, please use

    pytest tests --cov=src --cov-report=html --cov-report=term

and have a look at the `htmlcov` folder, after the tests are done.

### Notebooks

To use your module code (`src/`) in Jupyter notebooks (`notebooks/`) without running into import errors, make sure to install the source locally

    pip install -e .

This way, you'll always use the latest version of your module code in your notebooks via `import regression_comparison`.

Assuming you already have Jupyter installed, you can make your virtual environment available as a separate kernel by running:

    pip install ipykernel
    python -m ipykernel install --user --name="regression-comparison"

Note that we mainly use notebooks for experiments, visualizations and reports. Every piece of functionality that is meant to be reused should go into module code and be imported into notebooks.

### Distribution Package

To build a distribution package (wheel), please use

    python setup.py bdist_wheel

this will clean up the build folder and then run the `bdist_wheel` command.

### Contributions

Before contributing, please set up the pre-commit hooks to reduce errors and ensure consistency

    pip install -U pre-commit
    pre-commit install

If you run into any issues, you can remove the hooks again with `pre-commit uninstall`.

## Contact

- [thejonnyt](https://github.com/thejonnyt)
- [ossScharom](https://github.com/ossScharom)
- [philayzen](https://github.com/philayzen)
- [tomashkin](https://github.com/tomashkin)
- [pierreachkar1717](https://github.com/pierreachkar1717)

## License

© [WTFPL](./LICENCE)
