"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

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
import base64
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

data_intro_info = '''
The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43,943 tweets were collected. Each tweet is labelled as one of 4 classes, which are described below.'''

logistic = '''In statistics, the logistic model (or logit model) is a statistical model that models the probability of an event taking place by having the log-odds for the event be a linear combination of one or more independent variables. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (the coefficients in the linear combination). Formally, in binary logistic regression there is a single binary dependent variable, coded by an indicator variable, where the two values are labeled "0" and "1", while the independent variables can each be a binary variable (two classes, coded by an indicator variable) or a continuous variable (any real value).'''
d_tree = '''Decision tree learning is a supervised learning approach used in statistics, data mining and machine learning. In this formalism, a classification or regression decision tree is used as a predictive model to draw conclusions about a set of observations. Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Decision trees are among the most popular machine learning algorithms given their intelligibility and simplicity'''
rand = '''Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned.[1][2] Random decision forests correct for decision trees' habit of overfitting to their training set.[3]: 587–588  Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees'''
naive = '''In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features (see Bayes classifier). They are among the simplest Bayesian network models, but coupled with kernel density estimation, they can achieve high accuracy levels.'''
svc = '''The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the "predicted" class is. This makes this specific algorithm rather suitable for our uses, though you can use this for many situations.'''

def add_bg_from_local(image_file):
	with open(image_file, "rb") as image_file:
		encoded_string = base64.b64encode(image_file.read())
		st.markdown(f"""
		<style>
    	.stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    	}}
    	</style>
    		""",
    		unsafe_allow_html=True
    		)

	st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: deData.png;
    background-size: cover;
}
</style>
""",
    unsafe_allow_html=True,
)



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st. set_page_config(layout="wide")

	tab1, tab2, tab3 = st.tabs(['Home','Current Project','Twitter Classifier'])
	
	with tab1:
		add_bg_from_local('./resources/imgs/dashboard.png')
		with st.container():
			col1, col2, col3, = st.columns(3)
			with col2:
				st.markdown("""
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: top;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
				company_logo = Image.open('./resources/imgs/dimentionDataLogo.png')
				st.image(company_logo)



		with st.container():
			st.markdown("<h1 style='text-align: center; font-weight: 400; color: #ffdb59;'>WELCOME TO DIMENSION DATA</h1>", unsafe_allow_html=True)
			st.markdown('###')
			st.markdown('###')
	
		with st.container():
			col1,col2,col3 = st.columns(3)
			with col2:
				video1 = open("./resources/imgs/aboutus.mp4", "rb") 
				st.video(video1)

		with st.container():
			st.markdown('###')
			st.markdown('###')
			whoweare = Image.open('./resources/imgs/whoweare.png')
			st.image(whoweare)

		st.markdown('###')
		st.markdown('###')
		st.markdown("<h1 style='text-align: center; font-weight: 400; color: white;'>Our Trusted Partners</h1>", unsafe_allow_html=True)
		with st.container():
			col1,col2,col3 = st.columns(3)
			with col2:
				st.markdown('###')
				st.markdown('###')
				edsa = Image.open("./resources/imgs/EDSA_logo.png") 
				st.image(edsa)
	
	with tab2:
		st.markdown('###')
		st.markdown('###')
		st.markdown("<h1 style='text-align: center; color: #ffdb59;'>EDSA - Twitter Sentiment Classification</h1>", unsafe_allow_html=True)
		st.markdown('###')
		st.markdown('###')
		climatechange = Image.open("./resources/imgs/climatechange.png") 
		st.image(climatechange)
		st.markdown('###')
		st.markdown('###')
		st.markdown('###')
		st.markdown('###')
		st.markdown("<h1 style='text-align: center; color: #ffdb59;'>WHAT IS CLIMATE CHANGE?</h1>", unsafe_allow_html=True)
		st.markdown('###')
		st.markdown('###')
		st.markdown('###')
		st.markdown('###')

		col1, col2, col3 = st.columns(3)
		with col2:
			st.video('https://www.youtube.com/watch?v=EuwMB1Dal-4')
		st.markdown('###')
		st.markdown('###')

	with tab3:
		options = ["Data Information","Lets make a Prediction"]
		selection = st.sidebar.selectbox("Choose Option", options)

		# Building out the "Information" page
		if selection == "Data Information":
			col1, col2, col3 = st.columns(3)
			with col3:
				st.info("General Information")
			
			st.title('Where is this data from?')
			# You can read a markdown file from supporting resources folder
			#st.markdown(data_intro_info)
			st.markdown(f"<h3 style='text-align: center; color: #edff64 ;'>{data_intro_info}</h3>", unsafe_allow_html=True)
			st.markdown('###')
			st.markdown("<h4 style='text-align: center; color: #64FF76 ;'>NEWS\t\t- The tweet links to factual news about climate change.</h4>", unsafe_allow_html=True)
			st.markdown("<h4 style='text-align: center; color: #64FF76 ;'>PRO\t\t- The tweet supports the belief of man-made climate change.</h4>", unsafe_allow_html=True)
			st.markdown("<h4 style='text-align: center; color: #64FF76 ;'>NEUTRAL\t- The tweet neither supports nor refutes the belief of man-made climate change.</h4>", unsafe_allow_html=True)
			st.markdown("<h4 style='text-align: center; color: #64FF76 ;'>ANTI\t\t- The tweet does not believe in man-made climate change.</h4>", unsafe_allow_html=True)
			st.markdown('###')
			st.title('Machine Learning Models')
			st.markdown('###')
			st.markdown('###')
			#ption = st.selectbox('Available Machine Learning Models',('Logistic Regressor', 'Decision Tree', 'Random Forest','Naive Bayes','Linear SVC'))
			with st.container():
				col1, col2, col3 = st.columns(3)
				with col3:
					st.info("Logistic Regressor")
			st.markdown('###')
			st.markdown('###')
			
			st.markdown(f"<h5 style='text-align: center; color: #E4FFF2 ;'>{logistic}</h5>", unsafe_allow_html=True)
			st.markdown('###')
			st.markdown('###')
			st.markdown('###')
			with st.container():
				col1, col2, col3 = st.columns(3)
				with col3:
					st.info("Decision Tree")
					st.markdown('###')
			st.markdown('###')
			st.markdown('###')
			st.markdown(f"<h5 style='text-align: center; color:#E4FFF2 ;'>{d_tree}</h5>", unsafe_allow_html=True)
			st.markdown('###')
			st.markdown('###')
			st.markdown('###')
			
			# some info on logistic regressor
			with st.container():
				col1, col2, col3 = st.columns(3)
				with col3:
					st.info("Random Forest")
			st.markdown('###')
			st.markdown('###')
			st.markdown(f"<h5 style='text-align: center; color: #E4FFE4 ;'>{rand}</h5>", unsafe_allow_html=True)
			st.markdown('###')
			st.markdown('###')
			st.markdown('###')
			# some info on logistic regressor

			with st.container():
				col1, col2, col3 = st.columns(3)
				with col3:
					st.info("Naive Bayes")
			st.markdown('###')
			st.markdown('###')
			st.markdown(f"<h5 style='text-align: center; color: #FFFFE4 ;'>{naive}</h5>", unsafe_allow_html=True)
			# some info on logistic regressor

			with st.container():
				col1, col2, col3 = st.columns(3)
				with col3:
					st.info("Linear SVC")
			st.markdown('###')
			st.markdown('###')
			st.markdown(f"<h5 style='text-align: center; color: #E4FFF2 ;'>{svc}</h5>", unsafe_allow_html=True)
			st.markdown('###')
			st.markdown('###')
			st.markdown('###')

			
			st.subheader("Raw Twitter data and labels")
			if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				df = pd.read_csv("resources/train.csv")
			
				with st.container():
					st.dataframe(df)
				#st.write(raw[['sentiment', 'message']]) # will write the df to the page

		# Building out the predication page
		if selection == "Lets make a Prediction":

			col1, col2, col3 = st.columns(3)
			
			with col2:
				company_logo = Image.open('./resources/imgs/dimentionDataLogo.png')
				st.image(company_logo)
			st.markdown('###')
			st.markdown("<h1 style='text-align: center; color:  #FFE4E4 ;'>“As more and more artificial intelligence is entering into the world, more and more emotional intelligence must enter into leadership. - Amit Ray”</h1>", unsafe_allow_html=True)
			st.markdown('###')
			st.markdown("<h3 style='text-align: center; color:  #FFE4E4 ;'>Dimension Data has used NLP(Natural Language Proccessing) technique's along side a range of other amazing technologies to create a web application that is capable of classifying in what catagory a given tweet falls in. We have trained FIVE different models for you to pick from.</h3>", unsafe_allow_html=True)
			st.markdown('###')
			st.markdown('###')
			
			
			col1, col2, col3 = st.columns(3)
			class_description = ''
			file_name = ''
			with col2:
				st.subheader('Select a Machine Learning Model')
				option = st.selectbox('Available Machine Learning Models',('Logistic Regressor', 'Decision Tree', 'Random Forest','Naive Bayes','Linear SVC'))
			
			if option == 'Logistic Regressor':
				file_name = 'resources/Logistic_regression.pkl'

			elif option == 'Decision Tree':
				file_name = 'resources/Logistic_regression.pkl'

			elif option == 'Random Forest':
				file_name = 'resources/Logistic_regression.pkl'

			elif option == 'Naive Bayes':
				file_name = 'resources/Logistic_regression.pkl'

			elif option == 'Linear SVC':
				file_name = 'resources/Logistic_regression.pkl'
			
			tweet_text = st.text_area("","'Type your tweet in this here text box'")
			col1, col2, col3 = st.columns(3)
			with col2:
				if st.button("Classify"):
					# Transforming user input with vectorizer
					vect_text = tweet_cv.transform([tweet_text]).toarray()
					# Load your .pkl file with the model of your choice + make predictions
					# Try loading in multiple models to give the user a choice
					predictor = joblib.load(open(os.path.join(file_name),"rb"))
					prediction = predictor.predict(vect_text)
					class_description = ''
					if prediction == -1:
						class_description = 'Anti'
					if prediction == 0:
						class_description = 'Neutral'
					if prediction == 1:
						class_description = 'Pro'
					if prediction == 2:
						class_description = 'News'

					st.success("\tYour Tweet has been Categorized as : {}".format(class_description))
					st.balloons()
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			
		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
