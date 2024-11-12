# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import joblib
# import pickle
# from sklearn.preprocessing import LabelEncoder

# app = Flask(__name__)

# # Load the model and label encoder
# svc_model = joblib.load("app\\model\\apparel_size_predictor_model.pkl")
# with open('app\\model\\label_encoder_mapping.pkl', 'rb') as le_file:
#     label_classes = pickle.load(le_file)
# label_encoder = LabelEncoder()
# label_encoder.classes_ = label_classes

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Get data from form
#         height = float(request.form['height'])
#         weight = float(request.form['weight'])
#         chest = float(request.form['chest'])
#         waist = float(request.form['waist'])
#         hip = float(request.form['hip'])
#         size_purchased = request.form['size_purchased']

#         # Encode the 'Size Purchased' input if necessary
#         size_encoded = label_encoder.transform([size_purchased])[0]

#         # Prepare input array
#         input_data = np.array([[height, weight, chest, waist, hip, size_encoded]])

#         # Predict the recommended size
#         predicted_encoded_size = svc_model.predict(input_data)[0]
#         recommended_size = label_encoder.inverse_transform([predicted_encoded_size])[0]

#         return render_template('index.html', recommended_size=recommended_size)

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model and label encoder
svc_model = joblib.load("app\\model\\apparel_size_predictor_model.pkl")
with open('app\\model\\label_encoder_mapping.pkl', 'rb') as le_file:
    label_classes = pickle.load(le_file)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_size = None
    if request.method == 'POST':
        # Get data from form
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        chest = float(request.form['chest'])
        waist = float(request.form['waist'])
        hip = float(request.form['hip'])
        size_purchased = request.form['size_purchased']

        # Encode the 'Size Purchased' input
        size_encoded = label_encoder.transform([size_purchased])[0]

        # Prepare input array
        input_data = np.array([[height, weight, chest, waist, hip, size_encoded]])

        # Predict the recommended size
        predicted_encoded_size = svc_model.predict(input_data)[0]
        recommended_size = label_encoder.inverse_transform([predicted_encoded_size])[0]

    return render_template('index.html', recommended_size=recommended_size)

if __name__ == '__main__':
    app.run(debug=True)
