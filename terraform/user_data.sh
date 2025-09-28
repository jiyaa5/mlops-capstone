#!/bin/bash
# Update system and install Docker
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Pull and run your FastAPI container
docker pull jiyaa5/mlops-capstone:latest
docker run -d -p 8000:8000 jiyaa5/mlops-capstone:latest

# Optional: log running containers
docker ps > /home/ec2-user/docker_ps.log
