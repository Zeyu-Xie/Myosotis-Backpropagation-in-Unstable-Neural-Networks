# Src Simulation Iteration

模拟确定样本的迭代过程

## 用法

1. 构建数据库环境

```
docker run -d -p 27017:27017 --name localhost_mongodb -e MONGO_INITDB_ROOT_USERNAME=acan -e MONGO_INITDB_ROOT_PASSWORD=020301 mongo
```

2. 创建 Simulation_Recorder 数据库

3. 在 `main.py` 中修改 `sample_size` 为从 60000 个样本中抽取的样本数量

4. 运行（记得加参数）

```
python main.py [simulation_name]
```
