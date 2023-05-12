import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    url_for,
    send_from_directory,
)
import glob
import string
import shutil
import cv2

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

import random

import Dlib.d_lib as d_lib

# import DSFD.ds_fd as ds_fd
import Haar.HaarCascade as hc
import MT_CNN.mt_cnn as mt_c
import retinafaceresnet50.retinafaceresnet50 as rfr50
import retinanetmobilenetV1.retinanetmobilenetV1 as rmv1
import SSD.SSD as ssd
import YuNet.YuNet as yunet


import gc
import torch


def getabsPath(path):
    return os.path.abspath(path)


modelIndex = [
    "Dlib",
    # "DSFD",
    "Haar",
    "MT_CNN",
    "retinafaceresnet50",
    "retinanetmobilenetV1",
    "SSD",
    "YuNet",
]
PROJ_LOC = "LOCATION"

PROJ_CONST = PROJ_LOC.split("/")[-2]
PROJECT_ROOT = os.path.expanduser(PROJ_LOC)

data = {}
trainDir = getabsPath("./Training")
# models = [d_lib, ds_fd, hc, mt_c, rfr50, rmv1, ssd, yunet]
models = [d_lib, hc, mt_c, rfr50, rmv1, ssd, yunet]
# models = [d_lib, hc, rfr50, rmv1, ssd, yunet]


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"
db = SQLAlchemy(app)

userFile = []
userLocation = []
userDesc = []
caseNo = []


class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    casenumber = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    age = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    contact = db.Column(db.String(10), nullable=False)
    desc = db.Column(db.String(300), nullable=False)
    status = db.Column(db.String(15), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow())

    def __repr__(self) -> str:
        return f"<Person> {self.id}"


class Userinfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    casenumber = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(50), nullable=False)
    desc = db.Column(db.String(300), nullable=False)
    found = db.Column(db.String(500), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow())


with app.app_context():
    db.create_all()
    k = Person.query.all()
    for ind, val in enumerate(list(k)):
        data[ind + 1] = val.name + "=" + val.casenumber
    print(data)


# Set the upload folder and allowed extensions for uploaded files
app.config["UPLOAD_FOLDER"] = "Training"
app.config["USER_FOLDER"] = "userfiles"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

dt = []


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/registercase")
def registercase():
    return render_template("registercase.html")


@app.route("/infotable")
def infotable():
    k = Userinfo.query.order_by(Userinfo.date_created).all()

    return render_template("info.html", data=k)


@app.route("/infodel/<string:id>")
def infodel(id):
    print("hello", id)
    Userinfo.query.filter_by(casenumber=id).delete()
    db.session.commit()
    return redirect("/infotable")


@app.route("/inforeport/userfiles/<path:path>")
def images1(path):
    print("MAKSDK")
    return send_from_directory("userfiles", path)


@app.route("/inforeport/<string:id>")
def inforeport(id):
    _ID = []
    k = Userinfo.query.filter(Userinfo.casenumber == id).all()
    vv = k
    for i in vv:
        print("VV", i.desc)
        vv = i.desc
        break
    found = []
    tmp = []
    for i in k:
        tmp[:] = i.found.split(":")
        _ID.append(i.casenumber)
        break

    print(tmp)

    for i in tmp:
        k = i.split("=")
        name = k[0]
        id = k[1]
        k1 = Person.query.filter(Person.casenumber == id).all()
        print(k1)
        k2 = None
        for i in k1:
            k2 = i.id
            break
        found.append([name, k2])

    k = _ID.pop()

    print(found)
    # pr = os.getcwd() + f"/userfiles/{k}"

    ls = []

    for file in glob.glob(f"./userfiles/{k}/*"):
        # tmp = file.split("/")
        # for i in range(5):
        #     tmp.pop(0)
        # k = "/".join(tmp)
        ls.append(os.path.join(file))
        # print(file)

    p1 = ls[0:2]
    p2 = ls[2:4]
    p3 = ls[4:6]
    p4 = ls[6:8]

    return render_template(
        "inforeport.html", data=k, found=found, p1=p1, p2=p2, p3=p3, p4=p4, nd=vv
    )


def createCollage(name):
    def putName(loc, text):
        # Load the image
        img = cv2.imread(loc)
        os.remove(f"{text}.jpg")

        # Define the font and other properties of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 5
        color = (0, 255, 0)

        # Get the size of the text
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Put the text in the top left corner of the image
        x = 10
        y = text_size[1] + 10
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

        # Show the image
        cv2.imwrite(f"./{text}U.jpg", img)

    for file in glob.glob("./*.jpg"):
        name = file.split("/")[1].split(".")[0]
        putName(file, name)


@app.route("/testmodel", methods=["POST", "GET"])
def testClassifier():
    if request.method == "POST":
        try:
            ls = []
            getName = []

            os.chdir(PROJECT_ROOT)

            k = Person.query.all()
            for ind, val in enumerate(list(k)):
                data[ind + 1] = val.name + "=" + val.casenumber
            print(data)

            for ind, model in enumerate(models):
                torch.cuda.empty_cache()
                gc.collect()
                os.chdir(modelIndex[ind])
                getCwd = os.getcwd().split("/")
                getCwd.pop()
                newDir = "/".join(getCwd)
                ret = model.identify(data, trainDir, f"../{userFile[0]}")
                getName.append(userFile[0].split("/")[-1])
                k = glob.glob("*.jpg")

                src = os.path.abspath(k[0])
                dst = os.path.abspath("../") + "/" + userFile[0]

                dst = dst.split("/")
                dst.pop(-1)
                dst = "/".join(dst)

                shutil.move(src, dst)

                # os.system(f"cp {src} {dst}")
                if ret:
                    for i in ret.split(":"):
                        if i:
                            ls.append(i)
                os.chdir(newDir)

            ls = list(set(ls))
            dst = os.path.abspath("../") + "/" + f"{PROJ_CONST}" + userFile[0]

            dst = dst.split("/")
            dst.pop(-1)
            dst = "/".join(dst)
            os.chdir(f"{dst}")
            os.getcwd()
            createCollage(f"report-{caseNo[0]}")

            print(ls)

            if len(ls) > 0:
                print("ENTERD")
                new_task = Userinfo(
                    casenumber=caseNo[0],
                    location=userLocation[0],
                    desc=userDesc[0],
                    found=":".join(ls),
                )
                db.session.add(new_task)
                db.session.commit()

            os.chdir(PROJECT_ROOT)

            return "Tested"
        except Exception as e:
            raise e
    return render_template("facematching.html")


@app.route("/userinfo", methods=["POST", "GET"])
def indexv1():
    if request.method == "POST":
        os.chdir(PROJECT_ROOT)
        files = request.files.getlist("file")
        userLocation.append(request.form["location"])
        userDesc.append(request.form["desc"])
        caseNo[:] = []
        caseNo.append(
            "".join(
                random.SystemRandom().choice(string.ascii_uppercase + string.digits)
                for _ in range(10)
            )
        )
        ID = caseNo[0]
        filenames = []
        os.chdir(PROJECT_ROOT)
        for file in files:
            if file and allowed_file(file.filename):
                # Save the file to the upload folder
                filename = file.filename
                isExist = os.path.exists(
                    os.path.join(app.config["USER_FOLDER"] + "/" + ID)
                )
                if not isExist:
                    os.makedirs(os.path.join(app.config["USER_FOLDER"] + "/" + ID))
                file.save(os.path.join(app.config["USER_FOLDER"] + "/" + ID, filename))
                userFile[:] = []
                userFile.append(
                    os.path.join(app.config["USER_FOLDER"] + "/" + ID, filename)
                )
                filenames.append(filename)
        os.chdir(PROJECT_ROOT)

        return redirect("/testmodel")

    return render_template("infoprovider.html")


@app.route("/submitcase")
def submitCase():
    if request.method == "GET":
        if len(dt) != 0:
            return render_template("submitcase.html", dt=dt[0])
    return render_template("submitcase.html", dt="")


@app.route("/viewcases")
def viewcases():
    k = Person.query.order_by(Person.date_created).all()
    dat = {}

    for i in k:
        v = glob.glob(f"./Training/{i.id}/*")
        v = random.choice(v)
        dat[i.id] = os.path.join(v)
        # print(random.choice(v))
    return render_template("viewcases.html", data=k, dat=dat)


@app.route("/view/<int:id>")
def view(id):
    data = Person.query.get_or_404(id)
    dat = []

    v = glob.glob(f"./Training/{id}/*")
    for i in v:
        dat.append(os.path.join(i))

    return render_template("case.html", data=data, dat=dat)


@app.route("/found/<int:id>")
def found(id):
    data = Person.query.get_or_404(id)
    print(data)
    data.status = "FOUND"
    db.session.commit()

    return redirect("/viewcases")


@app.route("/Training/<path:path>")
def images(path):
    return send_from_directory("Training", path)


@app.route("/view/Training/<path:path>")
def imagesv1(path):
    return send_from_directory("Training", path)


@app.route("/trainmodel", methods=["POST"])
def trainClassifier():
    try:
        os.chdir(PROJECT_ROOT)
        k = Person.query.all()
        for ind, val in enumerate(list(k)):
            data[ind + 1] = val.name + "=" + val.casenumber
        print(data)

        for ind, model in enumerate(models):
            torch.cuda.empty_cache()
            gc.collect()
            os.chdir(modelIndex[ind])
            # os.system("	rm -rf model.yml startingPerson.txt *.jpg")
            getCwd = os.getcwd().split("/")
            getCwd.pop()
            newDir = "/".join(getCwd)
            # print(testImage)
            # model.identify(data, trainDir, testImage)
            model.identify(data, trainDir)
            os.chdir(newDir)
        return "Trained"
    except Exception as e:
        raise e


@app.route("/uploadinfo", methods=["POST"])
def upload():
    # Get the uploaded files from the request object
    files = request.files.getlist("file")
    name = request.form["personName"]
    age = request.form["age"]
    email = request.form["email"]
    number = request.form["number"]
    desc = request.form["desc"]
    casenumber = "".join(
        random.SystemRandom().choice(string.ascii_uppercase + string.digits)
        for _ in range(10)
    )
    status = "MISSING"
    ls = []

    if not name:
        dt[:] = []
        dt.append("name")
        return redirect("/submitcase")
    elif not age:
        dt[:] = []
        dt.append("Age")
        return redirect("/submitcase")
    elif not email:
        dt[:] = []
        dt.append("Email")
        return redirect("/submitcase")
    elif not number:
        dt[:] = []
        dt.append("Number")
        return redirect("/submitcase")
    elif not desc:
        dt[:] = []
        dt.append("Description")
        return redirect("/submitcase")
    elif files:
        for file in files:
            if file and allowed_file(file.filename):
                ls.append(file)
        if len(ls) == 0:
            dt[:] = []
            dt.append("Files")
            return redirect("/submitcase")
    print(name, age, email, number, desc, ls)
    new_task = Person(
        name=name,
        age=age,
        email=email,
        contact=number,
        desc=desc,
        status=status,
        casenumber=casenumber,
    )
    db.session.add(new_task)
    db.session.commit()

    obj = Person.query.all()
    ID = str(obj[-1].id)
    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            # Save the file to the upload folder
            filename = file.filename
            isExist = os.path.exists(
                os.path.join(app.config["UPLOAD_FOLDER"] + "/" + ID)
            )
            if not isExist:
                os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"] + "/" + ID))
            file.save(os.path.join(app.config["UPLOAD_FOLDER"] + "/" + ID, filename))
            filenames.append(filename)
    os.chdir(PROJECT_ROOT)

    return redirect("/submitcase")


if __name__ == "__main__":
    app.run(debug=True)
