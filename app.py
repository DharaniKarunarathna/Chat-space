from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import datetime


app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
socketio = SocketIO(app)
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)

# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.String(80), primary_key=True)
    password_hash = db.Column(db.String(120), nullable=False)

# Define Message model
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    text = db.Column(db.Text, nullable=True)
    image_path = db.Column(db.String(120), nullable=True)
    text_prediction = db.Column(db.String(200), nullable=True)  # New field
    image_prediction = db.Column(db.String(200), nullable=True)  # New field

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

@app.route('/')
@login_required
def index():
    messages = Message.query.all()
    return render_template('index.html', messages=messages)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.get(username):
            return 'User already exists!'
        hashed_password = generate_password_hash(password)
        new_user = User(id=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.get(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        return 'Invalid credentials!'
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if not current_user.id == 'admin':  # Ensure only admin can access
        return 'Access Denied', 403

    messages = []
    relevant_count = 0
    irrelevant_count = 0
    total_text_count = 0
    relevant_percentage = 0
    irrelevant_percentage = 0
    if request.method == 'POST':
        username = request.form['username']
        messages = Message.query.filter_by(username=username).all()
        for i in messages:
            print("text predicton : ",i.text_prediction)
            if i.text_prediction == 'irrelevant':
                irrelevant_count +=1
            elif i.text_prediction == 'relevant':
                relevant_count += 1
            else:
                pass

            if i.image_prediction == 'irrelevant':
                irrelevant_count +=1
            elif i.image_prediction == 'relevant':
                relevant_count += 1
            else:
                pass
        
        total_text_count = len(messages)
        print(total_text_count)

        # Calculate percentages
        if total_text_count > 0:
            relevant_percentage = (relevant_count / total_text_count) * 100
            print(relevant_count)
            irrelevant_percentage = (irrelevant_count / total_text_count) * 100
            print(irrelevant_count)



    return render_template('admin.html', 
                           messages=messages, 
                           relevant_percentage=relevant_percentage, 
                           irrelevant_percentage=irrelevant_percentage,
                           )




@socketio.on('message')
def handle_message(data):
    text = data.get('text')
    if text:
        predcictlable = textpredict(text)
        message = Message(username=current_user.id, text=text, text_prediction=predcictlable)
        db.session.add(message)
        db.session.commit()
    emit('response', data, broadcast=True)

@socketio.on('image')
def handle_image(data):
    filename = secure_filename(data['filename'])
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(data['file'])
    abpath = os.path.abspath(filepath)
    print("path of image for predict : ",abpath)
    predictimg = imagepredict(abpath)
    message = Message(username=current_user.id, image_path=filepath, image_prediction=predictimg)
    db.session.add(message)
    db.session.commit()
    
    emit('response', {'username': current_user.id, 'image': filepath}, broadcast=True)

@app.cli.command('initdb')
def initdb():
    db.create_all()
    print("Initialized the database.")


####################### ML #

# Load the trained model
modelimg = tf.keras.models.load_model('../final_model.keras')

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_image(modelimg, img_path):
    img_array = preprocess_image(img_path)
    predictions = modelimg.predict(img_array)
    return predictions

def interpret_prediction(predictions, threshold=0.5):
    if predictions[0] > threshold:
        return "irrelevant"
    else:
        return "relevant"
    
def imagepredict(imagepath):
    predictions = predict_image(modelimg, imagepath)
    class_prediction = interpret_prediction(predictions)
    print(class_prediction)
    return class_prediction

########### TEXT Predict ##############

# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('../text_model')
tokenizer = DistilBertTokenizer.from_pretrained('../text_tokenizer')

# Set the model to evaluation mode
model.eval()

# Move the model to the appropriate device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def predict(model, tokenizer, texts, device):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

def textpredict(text):
    # Example usage
    new_texts = ["This code not work well.", "let's go out."]
    predictions = predict(model, tokenizer, text, device)

    # Decode predictions
    predicted_labels = predictions.cpu().numpy()
    print(predicted_labels)
    if 0 in predicted_labels:
        return "irrelevant"
    else:
        return "relevant"
    


if __name__ == '__main__':
    socketio.run(app, debug=True)
