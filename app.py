from os import write

import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')

#loading models
clf=pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))


#Webapp
def main():
    st.title('Resume Screening App')

    #upload files either as text or pdf
    uploaded_file=st.file_uploader('Upload Resume',type=['text','pdf'])

    #Cleaning the input text function
    def clean_resume(txt):
        cleanTxt = re.sub(r'http\S+', ' ', txt)
        cleanTxt = re.sub(r'RT-cc\S+', ' ', cleanTxt)
        cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)
        cleanTxt = re.sub(r'#\S+', ' ', cleanTxt)
        special_chars = r"""!"$%&'()*+=-_./,:;<=>?@[\]^`{|}~"""
        escaped_special_chars = re.escape(special_chars)
        cleanTxt = re.sub(f'[{escaped_special_chars}]', ' ', cleanTxt)
        cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
        cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()
        return cleanTxt

    if uploaded_file is not None:
        try:
            resume_bytes=uploaded_file.read()
            resume_text=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text=resume_bytes.decode('latin-1')

        cleaned_resume=clean_resume(resume_text)
        input_features=tfidf.transform([cleaned_resume])
        prediction=clf.predict(input_features)[0]
        #mapping category to category name
        category_mapping = {
            6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing'
        }
        category_name=category_mapping.get(prediction,"Unknown")
        st.write("Predicted Category: ",category_name)

#python main
if __name__ == '__main__':
    main()