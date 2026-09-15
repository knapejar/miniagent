@echo off
rem miniagent driving Qwen3.8-27B on a Kaggle TPU (kaggle-tpu-lab). See README.
python "%~dp0miniagent.py" --kaggle %*
