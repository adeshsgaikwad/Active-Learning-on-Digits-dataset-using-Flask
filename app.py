from flask import *
import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
counter = int(1)
result = 1

n_initial = int(100)

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]

learner = ActiveLearner(
    estimator = RandomForestClassifier(),
    query_strategy = uncertainty_sampling,
    X_training = X_initial, y_training = y_initial
)
accuracy_scores = [learner.score(X_test, y_test)]

@app.route("/")
def get_form():
    return render_template('index.html')

@app.route("/post_field", methods=["GET", "POST"])
def post_field():
    if request.method == 'POST':
        global counter
        global result
        global accuracy_scores
        X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)
        query_idx, query_inst = learner.query(X_pool)
        if(counter == 1):
            result = request.form['this_name']
        else:
            guessedValue = request.form['guessed_value']
            y_new = np.array([int(guessedValue)], dtype=int)
            learner.teach(query_inst.reshape(1, -1), y_new)
            X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
            accuracy_scores.append(learner.score(X_test, y_test))

        result = int(result)

        if(counter <= result):
            query_idx, query_inst = learner.query(X_pool)
            fig = plt.figure(figsize=(10, 5))
            axis = fig.add_subplot(2, 2, 2)
            axis.set_title('Digit to label')
            axis.imshow(query_inst.reshape(8, 8))
            pngImage1 = io.BytesIO()
            FigureCanvas(fig).print_png(pngImage1)
            pngImageB64String1 = "data:image/png;base64,"
            pngImageB64String1 = base64.b64encode(pngImage1.getvalue()).decode('utf8')
            fig = plt.figure(figsize=(10, 5))
            axis = fig.add_subplot(2, 2, 2)
            axis.set_title('Accuracy of your model')
            axis.plot(range(counter), accuracy_scores)              # 1  THESE TWO LINES ARE CAUSING PROBLEMS
            axis.scatter(range(counter), accuracy_scores)           # 2
            axis.set_xlabel('number of queries')
            axis.set_ylabel('accuracy')
            pngImage2 = io.BytesIO()
            FigureCanvas(fig).print_png(pngImage2)
            pngImageB64String2 = "data:image/png;base64,"
            pngImageB64String2 = base64.b64encode(pngImage2.getvalue()).decode('utf8')
            counter = counter + 1
            return render_template("timepass.html", image1 = pngImageB64String1, image2 =pngImageB64String2)
        else:
            counter = int(1)
            fig = plt.figure(figsize=(10, 5))
            axis = fig.add_subplot(2, 2, 2)
            axis.set_title('Accuracy of the classifier during the active learning')
            axis.plot(range(result + 1), accuracy_scores)
            axis.scatter(range(result + 1), accuracy_scores)
            axis.set_xlabel('number of queries')
            axis.set_ylabel('accuracy')
            pngImage = io.BytesIO()
            FigureCanvas(fig).print_png(pngImage)
            pngImageB64String = "data:image/png;base64,"
            pngImageB64String = base64.b64encode(pngImage.getvalue()).decode('utf8')
            return render_template("final_result_page.html", image =pngImageB64String)

if __name__ == '__main__':
    app.run(debug = True)
