from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from starlette.responses import HTMLResponse
from packages.predict import predict, save_to_csv
from packages.preprocessing import preprocessing_csv, fusion_features, remove_tempfile
from fastapi.staticfiles import StaticFiles
import shutil
import os
import glob

# address: http://127.0.0.1:8000/docs
# launch :  uvicorn FastAPI:app --reload
# app.include_router(FastAPI.router, prefix='/app')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
result_path = "results/results.csv"


@app.post("/uploadtext")
async def upload_custom_csv(files: List[UploadFile] = File(...)):
    for file in files:
        if file.filename.endswith(".csv"):
            with open(file.filename, "wb") as f:
                f.write(file.file.read())
            if not os.path.exists('datas/' + file.filename):
                f.close()
                shutil.move(file.filename, 'datas/temp_test.csv')
            else:
                f.close()
                os.remove(file.filename)

        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted")

    return "Ok! c'est uploadé ! Retour à l'accueil: http://127.0.0.1:8000/index.html "


@app.post("/uploadimages/")
async def upload_custom_images(files: List[UploadFile] = File(...)):
    for file in files:
        valid_extensions = (".png", ".jpg", ".jpeg", ".JPG")
        if file.filename.endswith(valid_extensions):
            with open(file.filename, 'wb') as image:
                image.write(file.file.read())
            if not os.path.exists('datas/images/upload_images/' + file.filename):
                image.close()
                shutil.move(file.filename, 'datas/images/upload_images/')
            else:
                image.close()
                os.remove(file.filename)

        else:
            # Raise a HTTP 400 Exception, indicating Bad Request
            raise HTTPException(status_code=400, detail="Invalid file format. Only png, jpg, jpeg Files accepted. But "
                                                        "images files still be uploaded")

    return "Ok! c'est uploadé ! Retour à l'accueil: http://127.0.0.1:8000/index.html"


@app.get("/")
async def accueil():
    content = """
<head> 
<title>CognitivPyShop</title>
<link rel="shortcut icon" href="static/PICTOGRAMME.ico">
<link href='http://fonts.googleapis.com/css?family=Open+Sans' rel='stylesheet' type='text/css'>
<style>
html {
  font-size: 12px; 
  font-family: 'Open Sans', sans-serif; 
}
h2 {
  font-size: 25px;
  text-align: justify;
}
h3 {
  font-size: 15px;
  text-align: justify;
}
body {
  width: 700px;
  margin: 2px auto;
  padding: 2px 50px 50px 50px;
  border: 3px solid black;
}
img {
  display: block;
  margin: 0 auto;
}
body {
  background-image: url('https://denisbeauxarts.com/149649-home_default/sennelier-encre-flacon-30-ml-blanc-opaque.jpg');
  background-repeat: no-repeat;
  background-attachment: fixed;
  background-size: cover;
}
table, th, 
td {
    border: 2px solid #333;
    border-radius:10px;
    padding: 10px;
    border-spacing: 2px;
}

#input {
#  font-size: 1em;
#  padding-top: 0.35rem;
#}
</style>
</head> 
<body>
<img class="fit-picture"
     src="https://www.bigdataparis.com/2020/wp-content/uploads/sites/3/2019/07/Logo-Datascientest.png"
     alt="Fièrement propulsé par Datascientest">
<h2> Etape 1 : Upload des images pour le model </h2>
<h3> Formats acceptés : jpeg - jpg - png</h3>
<form action="/uploadimages/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<h2>Etape 2 : Upload du text pour le model </h2>
<h3>Formats acceptés : csv</h3>
<form action="/uploadtext/" enctype="multipart/form-data" method="post">
<input name="files" type="file">
<input type="submit">
</form>
<h2>Etape 3 : Lancement du model et des prédictions </h2>
<form action="/custom_predict", method="post">
<input type="submit" value="Prédictions" />
</form>
<h2> Etape 4 : Téléchargez vos résultats </h2>
<form action="/predictions", method="get">
<input type="submit" value="Résultats" />
</form>
<br>

</br>
<table>
  <tr>
    <th>Classe</th>
    <th>Correspondance</th>
  </tr>
  <tr>
    <td>0</td>
    <td>livres et romans</td>
  </tr>
  <tr>
    <td>1</td>
    <td>magazines</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Accessoires jeux videos</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Jouets enfance</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Livres et illustres</td>
  </tr>
  <tr>
    <td>5</td>
    <td>Papeteries</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Mobiliers jardin et cuisine</td>
  </tr>
  <tr>
    <td>7</td>
    <td>Mobiliers intérieurs et litteries</td>
  </tr>
  <tr>
    <td>8</td>
    <td>Jeux de société</td>
  </tr>
  <tr>
    <td>9</td>
    <td>Accessoires intérieurs</td>
  </tr>
  <tr>
    <td>10</td>
    <td>Livres jeunesse</td>
  </tr>
  <tr>
    <td>11</td>
    <td>Goodies geek</td>
  </tr>
  <tr>
    <td>12</td>
    <td>Piscine spa</td>
  </tr>
  <tr>
    <td>13</td>
    <td>Figurines Wargames</td>
  </tr>
  <tr>
    <td>14</td>
    <td>Modèles réduits ou télécommandes</td>
  </tr>
  <tr>
    <td>15</td>
    <td>Jeux geek</td>
  </tr>
  <tr>
    <td>16</td>
    <td>Cartes de collection</td>
  </tr>
  <tr>
    <td>17</td>
    <td>Décoration Intérieur</td>
  </tr>
  <tr>
    <td>18</td>
    <td>Jeux videos</td>
  </tr>
  <tr>
    <td>19</td>
    <td>Jeux et consoles retro</td>
  </tr>
  <tr>
    <td>20</td>
    <td>Petite enfance</td>
  </tr>
  <tr>
    <td>21</td>
    <td>Jouets enfants</td>
  </tr>
  <tr>
    <td>22</td>
    <td>Accessoires animaux</td>
  </tr>
  <tr>
    <td>23</td>
    <td>Jeux videos dematerialises</td>
  </tr>
  <tr>
    <td>24</td>
    <td>Jardin et bricolage</td>
  </tr>
  <tr>
    <td>25</td>
    <td>Epicerie</td>
  </tr>
  <tr>
    <td>26</td>
    <td>Matériel enfance</td>
  </tr>
</table>

<br>







</br>
<h3> Fièrement propulsé par Datasientest</h3>
</body>
    """
    return HTMLResponse(content=content)


# Create the POST endpoint with path '/predict'
@app.post("/predict")
async def upload_csv(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        # Create a temporary file with the same name as the uploaded
        # CSV file to load the data into a pandas Dataframe
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        file_pred = preprocessing_csv(file.filename)
        predictions = predict(file_pred)
        save_to_csv(file_pred, predictions)
        os.remove(file.filename)

        return "predictions sauvegardés !"
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")


@app.post("/custom_predict")
async def custom_predict():
    df = fusion_features()
    predictions = predict(df)
    save_to_csv(df, predictions)
    remove_tempfile()
    return "predictions sauvegardés !"


@app.get("/predictions")
def get_result():
    return FileResponse(path=result_path, media_type='text/csv', filename='predictions')

# if __name__ == "__main__":
#    uvicorn.run(FastAPI, port=8080, host='0.0.0.0', debug=True)
