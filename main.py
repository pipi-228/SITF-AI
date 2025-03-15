import os
from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import easyocr
from ultralytics import YOLO
import cv2
from pyaspeller import YandexSpeller

name="app" #idk for what need to work 

model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(name)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def render_page():
    return render_template("index.html", hide_image='hidden')

@app.route("/edit", methods=["GET", "POST"])
def edit():
    res_text = ""
    YoloPath = ""
    objects = ""

    speller = YandexSpeller()

    if request.method == 'POST':
        file = request.files['image']
        lang = request.form.get("lang_drop")
        
        if file and allowed_file(file.filename): 
            reader = easyocr.Reader([lang]) # install easyocr model

            filename = secure_filename(file.filename) # name of image file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # save image

            text_meta = reader.readtext("static/" + filename) # text recognition
            for (bbox, text, prob) in text_meta: # string formation
                res_text += text.upper() + " "
            fixed_res_text = speller.spelled(res_text) # text correction

            if os.path.exists('static/out.jpg'):
                os.remove('static/out.jpg')
            image_path = "static/" + filename
            image_meta = model(image_path)
            
            for result in image_meta:
                boxes = result.boxes
                for box in boxes:
                    obj_id = int(box.cls[0])
                    obj_name = str(result.names[obj_id])

                    objects += obj_name + " "

                    x1, y1, x2, y2 = map(int, box.xyxy[0]) #cords of rac
                    confidence = box.conf[0] #how much sure
                    class_id = int(box.cls[0]) # obj class
                    label = model.names[class_id] #class naming

                    #rendering
                    cv2.rectangle(result.orig_img, (x1-30, y1-30), (x2-30, y2-30), (0, 255, 0), 3)
                    cv2.putText(result.orig_img, f"{label} {confidence:.2f}", (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    
                    #resultsave
                    cv2.imwrite("static/out.jpg", result.orig_img)
                YoloPath="out.jpg" # for fix the always displayed prv variant

            return render_template("index.html", img_src=filename, YOLO=YoloPath, textedText=fixed_res_text, hide_image='', objs = objects)
        return "Неподдерживаемый тип файла. <a style='color: blue' onclick=window.history.back()>Вернуться</a>."

app.run(debug=True, port=5000)