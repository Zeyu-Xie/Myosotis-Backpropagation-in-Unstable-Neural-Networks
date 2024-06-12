#!/bin/bash

# Apt Settings
apt -y update
apt -y  upgrade
apt-get -y update
apt-get -y upgrade

# Install Node.js
# apt install -y curl
# curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
# apt-get install -y nodejs

# Install MongoDB Node.js Driver
# npm i mongodb

# Install Python3
apt install -y python3
apt install -y python3-pip

# Install Python3 Libraries
pip install -r /app/requirements.txt