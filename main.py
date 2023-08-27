from flask import Flask
import face_recognition
import pinecone
from PIL import Image
from flask import request
import requests as req
import numpy as np
import base64

app = Flask(__name__)
pinecone.init(api_key="a9d10522-c7d2-40f2-8bea-46793e13debe", environment="us-west4-gcp-free")
index = pinecone.Index("facerecog")

@app.route("/recognize",methods=['GET'])
def recognize():
    data = request.form
    imageBytes = data['image']
    imb64 = base64.b64decode(imageBytes)
    image = np.frombuffer(base64.b64decode(imb64),dtype=np.uint8)
    faceENC = face_recognition.face_encodings(image)[0]
    try:
        res = index.query(
            vector=list(list(faceENC)),
            top_k=1,
            include_values=True
        ).to_dict()['matches']
        return {"uid": res[0]['id']}
    except:
        return "Cannot find match!"

@app.route("/addFace",methods=['POST'])
def addFace(): 
    data = request.form
    uid = data['uid']
    imageBytes = data['image']
    imb64 = base64.b64decode(imageBytes)
    image = np.frombuffer(base64.b64decode(imb64),dtype=np.uint8)
    faceENC = face_recognition.face_encodings(image)[0]
    try:
        index.upsert([(uid,list(faceENC))])
        return "Added face successfully!"
    except Exception as e:
        return "Cannot add face!"

@app.route("/recognizeURL",methods=['GET'])
def recognizeURL():
    data = request.form
    imageURL = data['image']
    img = Image.open(req.get(imageURL,stream=True).raw)
    image = np.array(img).flatten()
    faceENC = face_recognition.face_encodings(image)[0]
    try:
        res = index.query(
            vector=list(list(faceENC)),
            top_k=1,
            include_values=True
        ).to_dict()['matches']
        return {"uid": res[0]['id']}
    except:
        return "Cannot find match!"

@app.route("/addFaceURL",methods=['POST'])
def addFaceURL(): 
    data = request.form
    uid = data['uid']
    imageURL = data['image']
    img = Image.open(req.get(imageURL,stream=True).raw)
    image = np.array(img).flatten()
    faceENC = face_recognition.face_encodings(image)[0]
    try:
        index.upsert([(uid,list(faceENC))])
        return "Added face successfully!"
    except Exception as e:
        return "Cannot add face!"

app.run(debug=True)