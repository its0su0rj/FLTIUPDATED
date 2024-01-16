import streamlit as st
import base64
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.corpus import stopwords
from io import BytesIO


import numpy as np
from PIL import Image
import pickle

from streamlit_option_menu import option_menu
nltk.data.path.append('nltk_data')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')


def predict_salary(loaded_model, input_features):
    loaded_model_salary = joblib.load('salary.joblib')

    try:
        # Convert input features to numeric values
        input_features = [float(feature) for feature in input_features]
    except ValueError:
        return "Please enter valid numeric values for all features."

    input_features = np.array(input_features).reshape(1, -1)

    # Predict the salary using the loaded model
    predicted_salary = loaded_model_salary.predict(input_features)[0]

    return predicted_salary


def diabetes_prediction(input_data):
    loaded_model_diabetes = joblib.load('diabetes.joblib')

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model_diabetes.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'



# Function for Loan Approval Prediction
def loan_approval_prediction(input_data):
    # Load the trained model for loan approval
    model_loan_approval = joblib.load('loan.joblib')

    # Convert categorical variables to numeric
    le = LabelEncoder()
    for column in input_data.keys():
        if isinstance(input_data[column], str):  # Check if the value is a string
            input_data[column] = le.fit_transform([input_data[column]])

    # Create a DataFrame with the input data
    input_df = pd.DataFrame(input_data)

    # Make predictions using the trained model
    predicted_loan_status = model_loan_approval.predict(input_df)

    return predicted_loan_status


# Function for Email Spam Detection
def email_spam_detection(input_text):
    # Load the trained model for email detection
    model_email_detection = joblib.load('email.joblib')

    # Make prediction using the loaded model
    prediction = model_email_detection.predict([input_text])

    return prediction


# Function for Image Compression
# ... (previous code)

# Function for Image Compression
# Function for Image Compression
def image_compression(input_image):
    # Load the trained model for image compression
    model_image_compression = joblib.load('image.joblib')

    # Preprocess the image
    input_image = np.array(input_image)

    # Ensure that the input image has 3 channels (RGB)
    if len(input_image.shape) == 2:
        input_image = np.stack([input_image] * 3, axis=-1)

    pixels = input_image.reshape(-1, 3)

    # Make predictions on the input data
    compressed_pixels = model_image_compression.cluster_centers_[model_image_compression.predict(pixels)]
    compressed_image = compressed_pixels.reshape(input_image.shape)

    return compressed_image

# Function for College Admission Probability
def college_admission_probability(input_data):
    # Load the trained model for college admission probability
    model_college_admission = joblib.load('admission.joblib')
    scaler = joblib.load('scalerad.joblib')
    # Standardize the features using the same scaler used during training
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the trained model
    predicted_admission_probability = model_college_admission.predict(input_data_scaled)

    return predicted_admission_probability[0]


# ... (previous code)


# Modify the ml_model_page function to include the new functions
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
        


def ml_model_page():
    set_png_as_page_bg('model.png') 
    st.title("ML Model Page")
    selected_tab = st.selectbox("Select Model", ["Loan Approval Prediction", "Email Spam Detection", "Image Compression", "College Admission Probability", "Diabetes Prediction", "Salary Prediction"])


    if selected_tab == "Loan Approval Prediction":
        st.write("You are on the Loan Approval Prediction page.")
        # Add specific content for Loan Approval Prediction
        #loan approval model
        
        
        # Streamlit UI
        st.title("Loan Approval Prediction")

        # Input form for user to enter data
        gender = st.selectbox("Select Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
        loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
        loan_amount_term = st.number_input("Loan Amount Term (Months)", min_value=0, step=1)
        credit_history = st.selectbox("Credit History", [1, 0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        # Create a DataFrame with the input data
        new_input = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area],
        }

        # Convert categorical variables to numeric
        le = LabelEncoder()
        for column in new_input.keys():
            if isinstance(new_input[column][0], str):  # Check if the value is a string
                new_input[column] = le.fit_transform(new_input[column])

        # Create a DataFrame with the new input data
        new_input_df = pd.DataFrame(new_input)

        # Make predictions using the loan approval model
        predicted_loan_status = loan_approval_prediction(new_input)

        # Display the predicted loan status
        if st.button("Predict Loan Approval Status"):
            if predicted_loan_status[0] == 1:
                st.success("Congratulations! Your loan is approved.")
            else:
                st.error("Sorry, your loan is not approved.")

    elif selected_tab == "Email Spam Detection":
        st.title("Email Spam Detection")

        # Input for user to enter an email
        user_input = st.text_area("Enter the email text:")

        # Button to trigger prediction
        if st.button("Check Spam"):
            # Make prediction using the email spam detection model
            prediction = email_spam_detection(user_input)

            # Display result
            if prediction[0] == 1:
                st.error("This email is classified as spam.")
            else:
                st.success("This email is not classified as spam.")

    elif selected_tab == "Image Compression":
        #n
        # Title
        st.title('Image Compression')
        
        # User input for image upload
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        
        if uploaded_file is not None:
            # Load and preprocess the image
            image = Image.open(uploaded_file)
            image = np.array(image)
        
            # Reshape the image to be a list of RGB values
            pixels = image_compression(image).reshape(-1, 3)
        
            # Make predictions on the input data
            model_image_compression = joblib.load('image.joblib')  # Load the model here
            compressed_pixels = model_image_compression.cluster_centers_[model_image_compression.predict(pixels)]
            compressed_image = compressed_pixels.reshape(image.shape)
    
            # Normalize pixel values to [0, 255]
            compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
        
            # Display the original and compressed images
            st.image(image, caption='Original Image', use_column_width=True)
            st.image(compressed_image, caption='Compressed Image', use_column_width=True)
        
            # Download button for the compressed image
            compressed_image_path = "compressed_image.jpg"
            compressed_image_bytes = BytesIO()
            Image.fromarray(compressed_image).convert("RGB").save(compressed_image_bytes, format="JPEG")
            st.download_button(
                label="Download Compressed Image",
                data=compressed_image_bytes.getvalue(),
                file_name=compressed_image_path,
                mime="image/jpeg"
            )
    elif selected_tab == "College Admission Probability":
        st.write("You are on the College Admission Probability page.")
        # Add specific content for College Admission Probability
        #college admission probability model
       
        # Streamlit UI
        st.title("College Admission Probability Prediction")

        # Input for user to enter data
        all_india_rank = st.number_input("Enter All India Rank:", min_value=1, max_value=1000, step=1)
        nirf_ranking = st.number_input("Enter NIRF Ranking:", min_value=1, max_value=30, step=1)
        placement_percentage = st.number_input("Enter Placement Percentage:", min_value=10, max_value=100, step=1)
        median_placement_package = st.number_input("Enter Median Placement Package (in lakhs):", min_value=10, max_value=200, step=1)
        distance_from_college = st.number_input("Enter Distance from College (in km):", min_value=10, max_value=2000, step=10)
        government_funded = st.checkbox("Is the College Government Funded?")
        teachers_qualification = st.number_input("Enter Teachers Qualification (1 to 10):", min_value=1, max_value=10, step=1)
        college_fee = st.number_input("Enter College Fee (in lakhs):", min_value=1, max_value=10, step=1)
        living_facilities = st.number_input("Enter Living Facilities (1 to 10):", min_value=1, max_value=10, step=1)
        girls_boys_ratio_percentage = st.number_input("Enter Girls/Boys Ratio Percentage (1 to 100):", min_value=1, max_value=100, step=1)

        # Create a DataFrame with the input data
        new_data = {
            'All_India_Rank': [all_india_rank],
            'NIRF_Ranking': [nirf_ranking],
            'Placement_Percentage': [placement_percentage],
            'Median_Placement_Package': [median_placement_package],
            'Distance_from_College': [distance_from_college],
            'Government_Funded': [1 if government_funded else 0],
            'Teachers_Qualification': [teachers_qualification],
            'College_Fee': [college_fee],
            'Living_Facilities': [living_facilities],
            'Girls_Boys_Ratio_Percentage': [girls_boys_ratio_percentage],
        }
     # Create a DataFrame with the input data
        #input_df = pd.DataFrame(new_data)

        # Make predictions using the college admission probability model
        #predicted_admission_probability = college_admission_probability(input_df)

        # Display the predicted admission probability
        #if st.button("Percentage Probability to Join the College"):
            #st.write(f"Predicted Admission Probability: {predicted_admission_probability:.2%}")

       
        input_df = pd.DataFrame(new_data)
        
        # Standardize the features using the same scaler used during training
        new_data_scaled = scaler.transform(input_df)
        #predicted_admission_probability = college_admission_probability(new_data_scaled)

        
        # Button to trigger prediction
        if st.button("Percentage Probability to Join the College"):
            # Make predictions using the trained model
            st.write(f"Input Features (scaled): {new_data_scaled}")
            predicted_admission_probability = college_admission_probability(new_data_scaled)
             #predicted_admission_probability = model.predict(new_data_scaled)
            
            st.write(f'Predicted Admission Probability: {predicted_admission_probability[0]}%')


    elif selected_tab == "Diabetes Prediction":
        st.write("You are on the Diabetes Prediction page.")
        # Add specific content for Diabetes Prediction
        #diabetes prediction model
        st.title('Diabetes prediction using SVM')

        # getting the input data from user
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose level')
        BloodPressure = st.text_input('Blood pressure value')
        SkinThickness = st.text_input('Skin thickness value')
        Insulin = st.text_input('Insulin level')
        BMI = st.text_input('BMI value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        Age = st.text_input('Age value')
        
        # code for prediction
        diagnosis = ''
        
        # creating a button for prediction
        if st.button('Diabetes test result'):
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                              DiabetesPedigreeFunction, Age])

        
        st.success(diagnosis)

        
        

    elif selected_tab == "Salary Prediction":
        st.write("You are on the Salary Prediction page.")
        # Add specific content for Salary Prediction
        #salary prediction model
        
        loaded_model_salary = joblib.load('salary.joblib')

            
        st.title('Salary CTC prediction using multiple linear regression')

        # getting the input data from the user
        grade = st.text_input('Grade (1-10)')
        softskill = st.text_input('Soft skill (1-10)')
        problemsolvingskill = st.text_input('Problem-solving skill (1-10)')
        meditationandyoga = st.text_input('Meditation and Yoga (1-10)')
        discipline = st.text_input('Discipline level (1-10)')
        strongcommandinoneskill = st.text_input('Strong command in one skill (1-10)')

        # code for prediction
        diagnosis = ''

        # creating a button for prediction
        if st.button('Predict Salary CTC in Lacs'):
            new_features = [grade, softskill, problemsolvingskill, meditationandyoga, discipline, strongcommandinoneskill]
            diagnosis = predict_salary(loaded_model_salary, new_features)

        st.success(f'Predicted Salary CTC: {float(diagnosis):.2f} Lacs' if isinstance(diagnosis, (int, float)) else diagnosis)

# Function for FLTI Page


# Function for Feedback Page
def feedback_page():
    set_png_as_page_bg('feedback1.png')  
    st.title("Feedback Page")
    st.write("You can contact me here:")
    st.write("Email - surajkumar@bhu.ac.in")
    st.write("For face to face meeting - Dalmia hostel, BHU")

    st.write("Or you can submit your feedback here:")
    feedback = st.text_area("Feedback")
    if st.button("Submit Feedback"):
        # Add logic to save feedback
        st.success("Thank you for your valuable time!")


def flti_page():
    set_png_as_page_bg('flti.png')  # Replace 'background.png' with your image file
    
    # Set the title with a specific text color
    st.markdown("<h1 style='color: #ff5733;'>FLTI</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: #ff5733;'>From Learning to Implementation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #0000ff;'>[Click on ML-MODEL](#)</p>", unsafe_allow_html=True)
    
    # Set the introductory text with a different text color
    st.markdown("<p style='color: #4287f5;'>Embark on a transformative journey with FLTI, where learning transcends theory into real-world impact. FLTI project seamlessly weaves algorithmic prowess with practical implementation, unlocking a realm of possibilities. Here's a glimpse into the brilliance of FLTI:</p>", unsafe_allow_html=True)
    
    # Add algorithm details with black color
    st.markdown("<p style='color: #000000;'><strong>Decision Tree Algo - Loan Approval Prediction:</strong><br>Unleash the power of decision trees to predict loan approval, navigating the intricate web of financial decisions with precision.</p>", unsafe_allow_html=True)

    st.markdown("<p style='color: #000000;'><strong>Naive Bayes - Email Spam Detection:</strong><br>Harnessing the simplicity of Naive Bayes, we've created an intelligent shield against inbox intruders, ensuring a spam-free digital sanctuary.</p>", unsafe_allow_html=True)

    st.markdown("<p style='color: #000000;'><strong>K-Means Clustering - Image Compression:</strong><br>Witness the magic of K-Means clustering as it compresses images, fusing efficiency with visual excellence in an algorithmic symphony.</p>", unsafe_allow_html=True)

    st.markdown("<p style='color: #000000;'><strong>KNN - College Admission Probability:</strong><br>Paving the way to higher education, KNN crafts a pathway by predicting college admission probabilities, guiding aspiring minds towards success.</p>", unsafe_allow_html=True)

    st.markdown("<p style='color: #000000;'><strong>Support Vector Machine - Diabetes Prediction:</strong><br>In the fight against diabetes, Support Vector Machine emerges as a formidable ally, predicting outcomes with accuracy and contributing to a healthier tomorrow.</p>", unsafe_allow_html=True)

    st.markdown("<p style='color: #000000;'><strong>Multiple Linear Regression - Salary Prediction:</strong><br>Chart your career trajectory with the precision of Multiple Linear Regression, foreseeing salary landscapes and empowering informed career decisions.</p>", unsafe_allow_html=True)

    # Add a line with red color
    st.markdown("<p style='color: #ff5733;'>its_su_rj</p>", unsafe_allow_html=True)


# Add authentication logic for the User page
def user_page():
    set_png_as_page_bg('login.png')  
    st.title("User Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Add authentication logic here
        st.success("Login successful!")
        # Add content for authenticated user



def main():
    with st.sidebar:
        selected = option_menu('select one',
                               ['FLTI', 'ML Model', 'User','Feedback'],
                               icons=['star','activity', 'heart', 'emoji-heart-eyes'],
                               default_index=0)
    if selected == "FLTI":
        flti_page()
    elif selected == "ML Model":
        ml_model_page()
    elif selected == "User":
        user_page()
    elif selected == "Feedback":
        feedback_page()
    # Create a horizontal layout
    col1, col2 = st.columns(2)

    with col1:
        # Content specific to the selected tab can be added here
        pass

if __name__ == "__main__":
    main()
