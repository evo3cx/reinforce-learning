Pre-requirements:

- python 3.9
- linux ubuntu 16.4 or later

install prequirements lib

```shell
sudo apt install python-opengl ffmpeg xvfb
```

install python dependencies
```shell
pip install -r requirements.txt
```

run train script

```shell
python taxy-qlearn.py
```

## Result

### Replay: 

<img alt="replay gih" src="https://github.com/evo3cx/reinforce-learning/blob/main/usecase/qlearn-taxi/replay.gif" />

### Metric:

```json
{
    "env_id": "Taxi-v3",
    "mean_reward": 7.56,
    "n_eval_episodes": 100,
    "eval_datetime": "2022-11-11 16:45:18"
}

```
