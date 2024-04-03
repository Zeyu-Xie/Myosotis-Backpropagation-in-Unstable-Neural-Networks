# Src

## Users

| Username            | Password | Roles           | Authentication Database |
| ------------------- | -------- | --------------- | ----------------------- |
| acan                | 020301   | root            | admin                   |
| acan_read_and_write | 020301   | readWrite@local | admin                   |

## Commands

1. Pull Official MongoDB Image from Docker Hub

```
docker pull mongo:latest
```

2. Run MongoDB Server & Define Root User

```
docker run -v .:/app -d -p 27017:27017 --name localhost_mongodb -e MONGO_INITDB_ROOT_USERNAME=acan -e MONGO_INITDB_ROOT_PASSWORD=020301 mongo
```

3. Rerun MongoDB Container

```
docker restart localhost_mongodb
```

4. Execute Commands in the Container

```
docker exec -it localhost_mongodb /bin/bash
mongosh -u acan -p 020301
```

or you can directly enter mongosh instead of bash

```
docker exec -it localhost_mongodb mongosh -u acan -p 020301
```

5. Stop MongoDB Container

```
docker stop localhost_mongodb
```

6. Remove MongoDB Container

```
docker rm localhost_mongodb
```
