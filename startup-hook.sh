#!/bin/bash
# 出错自动终止脚本
set -e

python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# download Arial.ttf
mkdir -p /root/.config/Ultralytics
cp models/Arial.ttf /root/.config/Ultralytics
cp models/Arial.Unicode.ttf /root/.config/Ultralytics