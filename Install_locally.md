# Install Ray and required libraries locally

Here, you can find instructions about how to locally install Ray and all libraries required for this workshop.

Follow the instructions to set it up.

## Clone this repository

In your terminal run:

```
git clone https://github.com/ray-project/llms-in-prod-workshop-2023.git
```

Go to the repository directory. In your terminal run:

```
cd llms-in-prod-workshop-2023
```

## Instructions for Unix users

You should install dependencies in a new virtual environment. If you are not familiar with virtualenv, consult their [documentation](https://docs.python.org/3/library/venv.html) before you move forward.

Make sure that you have virtual env with `python==3.10.10`.

Once in the virtualenv, run in your terminal:

```
pip install -r requirements.txt
```

Make sure to install `torch==1.13.1` that is suitable for your system (CPU/GPU and CUDA version). See PyTorch get started [documentation](https://pytorch.org/get-started/locally/) for more details.

## Test an environment

Check if Ray is installed correctly.

### Start Python in the interactive mode

In your terminal run:

```
python3
```

Wait until you see `>>>` prompt.

### Import ray

```
>>> import ray
```

### Start Ray runtime locally

```
>>> ray.init()
```

If you see output like this, Ray is installed correctly:

```
2022-12-07 11:15:08,106 INFO worker.py:1519 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265
```

Feel free to open Ray dashboard at [http://127.0.0.1:8265](http://127.0.0.1:8265)

# What to do next?

Go to [README](README.md) to get started.

----

# Troubleshooting
Check Ray installation [documentation](https://docs.ray.io/en/latest/ray-overview/installation.html) for more details and [troubleshooting](https://docs.ray.io/en/latest/ray-overview/installation.html#troubleshooting) for issues with installing.

<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Generic/ray_logo.png" width="30%" loading="lazy">
