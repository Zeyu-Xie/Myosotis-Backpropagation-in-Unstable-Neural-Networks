#!/bin/zsh

# 循环执行命令，参数从 0 到 99
for i in {0..98}; do
    echo "python ./main.py $i"
    python ./main.py $i
done
