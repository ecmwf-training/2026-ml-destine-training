# Anemoi-Training

This README contains the instructions for running deterministic training of a data-driven weather forecasting model.

## Seed

First we need to export a SEED that controls the randomness of the training

```bash
export ANEMOI_BASE_SEED=42
```

## Training 

Now we start training
```bash
anemoi-training train --config-name=deterministic_minimal.yaml
```

## Logging

Open a new terminal tab and start the mlflow local server
```
mlflow ui --backend-store-uri=/home/$USER/anemoi-output/logs/mlflow/ --port=5111
```

Open a new tab and go to

```
https://traininglab02.ecmwf.europeanweather.cloud/user/[USER_ID_GOES_HERE]/proxy/5111/
```