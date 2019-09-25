# Getting Started

## Installation

### Latest Stable Version (pip installation):
``` bash
pip install imageatm
```

### Bleeding Edge Version (manual installation):
``` bash
git clone https://github.com/idealo/imageatm.git
cd imageatm
python setup.py install
```

### Folder structure and file formats
The starting folder structure is
```
root
├── config_file.yml
├── data.json
└── images
    ├── image_1.jpg
    ├── image_2.jpg
    ├── image_3.jpg
    └── image_4.jpg

```
`data.json` is a file containing a mapping between the images and their labels. This file must be in the following format:

``` json
[
    {
        "image_id": "image_1.jpg",
        "label": "Class 1"
    },
    {
        "image_id": "image_2.jpg",
        "label": "Class 1"
    },
    {
        "image_id": "image_3.jpg",
        "label": "Class 2"
    },
    {
        "image_id": "image_4.jpg",
        "label": "Class 2"
    },
    ...
]

```

In the next sections we will use the cats and dogs dataset to showcase all examples. Therefore our starting structure will look as follows:
```
root
 ├── data.json
 ├── cats_and_dogs_job_dir
 └── cats_and_dogs
     └── train
         ├── cat.0.jpg
         ├── cat.1.jpg
         ├── dog.0.jpg
         └── dog.1.jpg  
```

Here you can download the cats and dogs dataset:
``` bash
wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O cats_and_dogs_filtered.zip

unzip cats_and_dogs_filtered.zip

mkdir -p cats_and_dogs/train
mv cats_and_dogs_filtered/train/cats/* cats_and_dogs/train
mv cats_and_dogs_filtered/train/dogs/* cats_and_dogs/train
```

Convert the data into our input format (`data.json`):
``` json
[
    {
        "image_id": "cat.0.jpg",
        "label": "Cat"
    },
    {
        "image_id": "cat.1.jpg",
        "label": "Cat"
    },
    {
        "image_id": "dog.0.jpg",
        "label": "Dog"
    },
    {
        "image_id": "dog.1.jpg",
        "label": "Dog"
    },
    ...
]
```

You can use this code to create the `data.json`:
``` python
import os
import json

filenames = os.listdir('cats_and_dogs/train')
sample_json = []
for i in filenames:
    sample_json.append(
        {
        'image_id': i,
        'label': 'Cat' if 'cat' in i else 'Dog'
        }
        )

with open('data.json', 'w') as outfile:
    json.dump(sample_json, outfile, indent=4, sort_keys=True)
```
## Simple Example for local training
### Train with CLI
Define your `config_file.yml`:
``` yaml
image_dir: cats_and_dogs/train
job_dir: cats_and_dogs_job_dir/

dataprep:
  run: True
  samples_file: data.json
  resize: True

train:
  run: True

evaluate:
  run: True
```

These configurations will run three components in a pipeline: data preparation, training, and evaluation.

Then run:
```
imageatm pipeline config/config_file.yml
```

The resulting folder structure will look like this
```
root
├── config_file.yml
├── data.json
├── cats_and_dogs
│   └── train
│       ├── cat.0.jpg
│       ├── cat.1.jpg
│       ├── dog.0.jpg
│       └── dog.1.jpg
└── job_dirs
    └── cats_and_dogs_job_dir
        ├── class_mapping.json
        ├── test_samples.json
        ├── train_samples.json
        ├── val_samples.json
        ├── logs
        ├── models
        └── evaluation
```

### Train without CLI
Run the data preparation:
``` python
from imageatm.components import DataPrep

dp = DataPrep(
    image_dir = 'cats_and_dogs/train',
    samples_file = 'data.json',
    job_dir='cats_and_dogs_job_dir'
)

dp.run(resize=True)
```

Run the training:
``` python
from imageatm.components import Training

trainer = Training(dp.image_dir, dp.job_dir)
trainer.run()
```

Run the evaluation:
``` python
from imageatm.components import Evaluation

evaluater = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)
evaluater.run()
```

## Simple Example for cloud training
### Initial cloud set-up
To train your model using cloud services you'll need an existing S3 bucket where you will be able to store the content of your local job_dir and image_dir as well as trained models.

If you don't have an S3 bucket you'll have to [create one](https://docs.aws.amazon.com/quickstarts/latest/s3backup/step-1-create-bucket.html).

#### Assign the following **IAM roles** to your AWS user account

* iam role:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "Stmt1508403745000",
            "Effect": "Allow",
            "Action": [
                "iam:CreatePolicy",
                "iam:CreateRole",
                "iam:GetPolicy",
                "iam:GetRole",
                "iam:GetPolicyVersion",
                "iam:CreateInstanceProfile",
                "iam:AttachRolePolicy",
                "iam:ListRolePolicies",
                "iam:GetInstanceProfile",
                "iam:ListEntitiesForPolicy",
                "iam:ListPolicyVersions",
                "iam:CreatePolicyVersion",
                "iam:RemoveRoleFromInstanceProfile",
                "iam:DetachRolePolicy",
                "iam:DeleteInstanceProfile",
                "iam:DeletePolicyVersion",
                "iam:ListInstanceProfilesForRole",
                "iam:DeletePolicy",
                "iam:DeleteRole",
                "iam:AddRoleToInstanceProfile",
                "iam:PassRole"
            ],
            "Resource": [
                "*"
            ]
        }
    ]
}
```

* ec2 role:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": "ec2:*",
            "Effect": "Allow",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "elasticloadbalancing:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "cloudwatch:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "autoscaling:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "iam:CreateServiceLinkedRole",
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "iam:AWSServiceName": [
                        "autoscaling.amazonaws.com",
                        "ec2scheduled.amazonaws.com",
                        "elasticloadbalancing.amazonaws.com",
                        "spot.amazonaws.com",
                        "spotfleet.amazonaws.com",
                        "transitgateway.amazonaws.com"
                    ]
                }
            }
        }
    ]
}
```

* s3 role:
``` json
{
   "Version": "2012-10-17",
   "Statement": [
       {
           "Effect": "Allow",
           "Action": "s3:*",
           "Resource": "*"
       }
   ]
}
```

#### Configure your AWS client

In your shell run `aws configure` and input:

    - AWS Access Key ID
    - AWS Secret Access Key
    - Default region name (for example `eu-central-1`)
    - Default output format (for example `json`)

Now you are ready to kick off with the cloud training.

### Train with CLI

Define your `config_file.yml`:
``` yaml
image_dir: cats_and_dogs/train
job_dir: cats_and_dogs_job_dir/

dataprep:
  run: True
  samples_file: data.json
  resize: True

train:
  run: True
  cloud: True

cloud:
  cloud_tag: image_atm_example_tag
  provider: aws
  tf_dir: cloud/aws
  region: eu-west-1
  vpc_id: vpc-example
  instance_type: p2.xlarge
  bucket: s3://example-bucket
  destroy: True
```
These configurations will **locally** run data preparation, then launch an AWS EC2 instance, run a training on it and copy the content of your local job_dir and image_dir to the pre-defined S3 bucket.
As we set `destroy: True` in this example, it will also destroy the EC2 instance and all dependencies once the training finished. If you want to reuse the
instance for multiple experiments, set `destroy: False`.

Run imageatm in shell:
``` bash
imageatm pipeline config/config_file.yml
```

The resulting **S3 bucket** structure will look like this:
```
s3://example-bucket
├── image_dirs
│   └── train_resized
│       ├── cat.0.jpg
│       ├── cat.1.jpg
│       ├── dog.0.jpg
│       └── dog.1.jpg
└── job_dirs
    └── cats_and_dogs_job_dir
        ├── class_mapping.json
        ├── test_samples.json
        ├── train_samples.json
        ├── val_samples.json
        └── models
            ├── model_1.hdf5
            ├── model_2.hdf5
            └── model_3.hdf5
```
The resulting **local** structure will look like this:
```
root
├── cats_and_dogs
│   ├── train
│   │   ├── cat.0.jpg
│   │   ├── cat.1.jpg
│   │   ├── dog.0.jpg
│   │   ├── dog.1.jpg
│   └── train_resized
│       ├── cat.0.jpg
│       ├── cat.1.jpg
│       ├── dog.0.jpg
│       └── dog.1.jpg
│
└── cats_and_dogs_job_dir
    ├── class_mapping.json
    ├── test_samples.json
    ├── train_samples.json
    ├── val_samples.json
    ├── logs
    └── models
        ├── model_1.hdf5
        ├── model_2.hdf5
        └── model_3.hdf5
```

### Train without CLI

Make sure you've got your cloud setup ready as described in the [cloud training introduction](#initial-cloud-set-up).

Run the data preparation:
``` python
from imageatm.components import DataPrep

dp = DataPrep(
    image_dir = 'cats_and_dogs/train',
    samples_file = 'data.json',
    job_dir='cats_and_dogs_job_dir'
)

dp.run(resize=True)
```

Run the training:
``` python
from imageatm.components import AWS

cloud = AWS(
    tf_dir='cloud/aws',
    region='eu-west-1',
    instance_type='p2.xlarge',
    vpc_id='vpc-example',
    s3_bucket='s3://example-bucket',
    job_dir=dp.job_dir,
    cloud_tag='image_atm_example_tag',
)

cloud.init()
cloud.apply()
cloud.train(image_dir=dp.image_dir)
```

Once the training is completed you have to manually destroy the AWS instance and its dependencies:
``` python
cloud.destroy()
```
