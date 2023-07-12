#! /bin/bash

python_version=`python --version`

if [ "$str1" != "Python 3.11.3" ]; then
    echo "Warning! You are not using the recommended python version for this app"
    echo "Recommended python version: Python 3.11.3"
    echo "Current python verison: $python_version"
    echo "Trying to install the modules with the the not recommended version"
    sleep 5
fi

python -m venv env
source env/bin/activate

pip install -r requirements.txt
pip install -e .

echo "Starting Regression Education"
python src/regression_edu/apps/regression_education.py