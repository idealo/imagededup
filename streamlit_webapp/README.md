# 🗣 Imagededup - Streamlit based Web App 🚀

A simple streamlit based web application in order to find duplicates in a corpus of images using perceptual hashing, uploaded by the user.

![demo](https://user-images.githubusercontent.com/29462447/203412429-ebdcc031-dd6e-4f32-87df-69f1931d75e1.gif)


## Installation:
* Simply run the command ***pip install -r requirements.txt*** in order to install the necessary dependencies.

## Usage:
1. Simply run the command: 
```
streamlit run app.py
```
3. Navigate to http://localhost:8501 in your web-browser. This will launch the web app :

![1](https://user-images.githubusercontent.com/29462447/203628192-337df35e-012a-49c6-99c0-e9de43c70d5f.png)


4. By default, streamlit allows us to upload files of **max. 200MB**. If you want to have more size for uploading images, execute the command :
```
streamlit run app.py --server.maxUploadSize=1028
```

### Running the Dockerized App
1. Ensure you have Docker Installed and Setup in your OS (Windows/Mac/Linux). For detailed Instructions, please refer [this.](https://docs.docker.com/engine/install/)
2. Navigate to the folder where you have cloned this repository ( where the ***Dockerfile*** is present ).
3. Build the Docker Image (don't forget the dot!! :smile: ): 
```
docker build -f Dockerfile -t app:latest .
```
4. Run the docker:
```
docker run -p 8501:8501 app:latest
```

This will launch the dockerized app. Navigate to ***http://localhost:8501/*** in your browser to have a look at your application. You can check the status of your all available running dockers by:
```
docker ps
```



