# Bayesian Neural Controlled Differential Equations for Treatment Effect Estimation


1) First, install requirements:
```
pip install -r requirements.txt
```

2) Training of BNCDE / TE-CDE:

Open port for mlfow via

```
mlflow server --port=5002
```

Start the training run of BNCDE through
```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> python3 src/train_scripts/bncde_train_call.py
```

Start the training run of TE-CDE with dropout through
```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> python3 src/train_scripts/tecde_train_call.py
```

Start the training run of both models under informative sampling via
```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> python3 src/train_scripts/SAR_train_call.py
```


To connect the web UI via ssh, enter:
```
ssh -N -f -L localhost:5002:localhost:5002 <username>@<server-link>
```
Access the web UI through the browser http://localhost:5002.



3) To run the experiments, specify the paths of the BNCDE and TE-CDE models, the prediction window and whether informative sampling is turned on in 

```
Code/config/experiments.yml
```

Run the experiments via
```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> python3 src/test_scripts/results_call.py
```

The results are then stored in
```
data/results
```

The plotting functions can be found in 
```
src/test_scripts/results_plot.py
```
and can be executed locally. The figures are stored in the figures directory.

