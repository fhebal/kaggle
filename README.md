# kaggle
models and deployments for the kaggle projects, e.g. classification, regression, and clustering

# repo stuff
I'm using https://www.conventionalcommits.org/en/v1.0.0/ for commits to keep it simple
I'm loosely using https://drivendata.github.io/cookiecutter-data-science/ to keep things organized

# modeling
I'm doing this https://www.kaggle.com/code/pliptor/how-am-i-doing-with-my-score to learn everything i can with all different frameworks (Tensorflow, sklearn, etc..) take a look at each project folder to see what's been done, its documented for each in its own README
I'm trying local cpu in codespaces, but also connecting to like my local machine using a NVIDEO 3070 GPU 8GB
I will try some other stuff if the cost isn't to crazy

# pipelines
I'm doing pachyderm cause its free
Retraining triggers and such when drift is detected among other things

# CICD
github actions
black
flake
maybe some other stuff

# tests
data typing
integration test
load testing for api endpoints

# deployment
I'm hosting my own apis and doing deployent with docker images on flask or fastapi or soemthign and on my local home machine
I'm hosting the apps in aws for practice deploying