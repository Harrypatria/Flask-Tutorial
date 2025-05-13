 import numpy as np
 from flask import Flask, request, jsonify, render_template
 # Flask - To import flask,request - Getting the request data (for which predictions are 
to be made), 
# jsonify - jsonify our predictions and send the response back
 import pickle 
app = Flask(__name__) #create an instance of flask.
 model = pickle.load(open('model.pkl', 'rb')) #Load our model pickle file 
@app.route('/')
 def home(): 
return render_template('index.html') 
# @app.route('/') is used to tell flask what url should trigger the function index() 
# and in the function index we use render_template('index.html') to display the script
 index.html in the browser.
 @app.route('/predict',methods=['POST'])
 """ Let's write a function predict() which will do: 
1) Load the persisted model into memory when the application starts,
 2) Create an API endpoint that takes input variables, transforms them into the appropri
 ate format, and returns predictions.""" 
def predict(): 
'''
   For rendering results on HTML GUI
   ''' 
int_features = [int(x) for x in request.form.values()] #Take Input as integer value
 s 
final_features = [np.array(int_features)] #convert it into array 
prediction = model.predict(final_features) #PRedict 
output = round(prediction[0], 2) 
return render_template('index.html', prediction_text='Profit should be $ {}'.format
 (output))  
@app.route('/predict_api',methods=['POST'])
 def predict_api(): 
'''
   For direct API calls trought request
   ''' 
data = request.get_json(force=True) 
prediction = model.predict([np.array(list(data.values()))]) 
output = prediction[0] 
return jsonify(output) 
if __name__ == "__main__": 
app.run(debug=True)
