# NLP-Name-Entity-Recognition
This is an NLP Project for Token Classification also known as Name Entitiy Recognition


## Workflows

 - constants
 - config_entity
 - artifact_entity
 - components
 - pipeline
 - app.py


## Git commands

```bash
git add .

git commit -m "Updated"

git push origin main
```


## AWS Configuration

- Run the below commands in the prompt

```bash
aws configure
```

## How to run?

```bash
conda create -n nerproj python=3.8 -y
```

```bash
conda activate name_entity_recognition
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```


### How to run?

```bash
conda create -n object-detection-industry-safety-checks python=3.10.14 -y
```

```bash
conda activate object-detection-industry-safety-checks
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

# AWS-CICD-Deployment-with-Github-Actions

### 1. Login to AWS console.

### 2. Create IAM user for deployment with specific access

	1. AmazonEC2ContainerRegistryFullAccess
	2. AmazonEC2FullAccess

	1. EC2 access : It is virtual machine
	2. ECR: Elastic Container registry to save your docker image in aws

### 3. How to setup Application on EC2: About the deployment

	1. Build docker image of the source code
	2. Push your docker image to ECR
	3. Launch Your EC2
	4. Pull Your image from ECR in EC2
	5. Lauch your docker image in EC2


### 4. Create ECR repo to store/save docker image
- Save the URI: <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/<ECR_REPOSITORY_NAME>


### 5. Create EC2 machine (Ubuntu)

### 6. Open EC2 and Install docker in EC2 Machine:

- Optinal

```bash
sudo apt-get update -y
sudo apt-get upgrade
```
	
- Required

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 7. Configure EC2 as self-hosted runner:
- setting>actions>runner>new self hosted runner> choose os> then run command one by one

### 8. Setup github secrets:

- Setup the values for aws credentials in the Prompt:
    - AWS_ACCESS_KEY_ID=
    - AWS_SECRET_ACCESS_KEY=
    - AWS_REGION = us-east-1
    - AWS_ECR_LOGIN_URI = <AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/<ECR_REPOSITORY_NAME>
    - ECR_REPOSITORY_NAME
