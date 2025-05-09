"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("vectorizer.pkl","rb")
test_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
#raw = pd.read_csv("streamlit/train.csv")


def read_markdown_file(markdown_file):
    	return Path(markdown_file).read_text()

# The main function where we will build the actual app
def main():
	"""News Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("News Classifer")
	st.subheader("Analysing news articles")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information","Model Stats"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Specify your markdown file name/path
	markdown_file = '../README.md'  
	markdown_content = read_markdown_file(markdown_file)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown(markdown_content)

	# Created a list of images to be shown once a prediction is made and their paths 
	images ={
	"business": "../Images/Business.jpg",
	"education": "../Images/Education.jpg",
	"entertainment": "../Images/Entertainment.jpg",
	"sports": "../Images/Sports.jpg",
	"technology": "../Images/technology.jpg"
	}

	# Created a list of models and their paths to the saved models
	models = {
    "SVC": "SVM.pkl",
	"LogisticRegression": "LogisticRegression.pkl",
    "Naive_Bayes": "Naive_Bayes.pkl"}

	model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
		
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		news_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			#vectorizer = TfidfVectorizer()
			#X_vectorized = vectorizer.fit_transform([news_text]) 
			vect_text = test_cv.transform([news_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("streamlit/Logistic_regression.pkl"),"rb"))
			predictor = joblib.load(open(models[model_choice], "rb"))

			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#st.success("Text Categorized as: {}".format(prediction))
			st.success(f"Text Categorized as: {prediction[0]} using {model_choice}")
			st.image(images[prediction[0]], caption='', use_container_width=True)
			#if hasattr(predictor, "predict_proba"):
    			#proba = predictor.predict_proba(vect_text)
   				#st.write("Prediction Probabilities:", proba)



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
