from flask import Flask, request
import pickle

app = Flask(__name__)

model = pickle.load(open("iris_logistic_model.pkl", "rb"))

@app.route("/prediction", methods=["POST"])
def prediction():
    sl = float(request.form["sl"])
    sw = float(request.form["sw"])
    pl = float(request.form["pl"])
    pw = float(request.form["pw"])

    result = model.predict([[sl, sw, pl, pw]])

    return str(result[0])

if __name__ == "__main__":
    app.run(debug=True)
