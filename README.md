# NLP-Project
📁 NLP-PROJECT 

│  

├── app.py # Streamlit interface   

├── CODES/ # Modular ML pipeline scripts 

 ├── visualizations/ # PCA, t-SNE, confusion matrix 

 ├── requirements.txt # Dependencies 

├── predict_realtime.py # Real Time voice Translator using the system mic 

 └── README.md # Documentation 

 

MODEL: https://huggingface.co/vascure/DogVoiceTranslatorModel/tree/main 
 
 
Installation Guide 

1️⃣ Clone the repository 

git clone https://github.com/JyotsnaPandranki/NLP-Project 
cd NLP-Project 
 

2️⃣ Create Virtual Environment 

python3.10 -m venv venv 
source venv/bin/activate 
 

3️⃣ Install system-level dependencies 

brew install portaudio ffmpeg 
 

4️⃣ Install Python dependencies 

pip install -r requirements.txt 

 

NOTE: DOWNLOAD MODELS : 
https://huggingface.co/vascure/DogVoiceTranslatorModel/tree/main 

 

Run the App 

streamlit run app.py 

 

 

 

Dog Voice Translation Using Deep Learning 

A deep learning system that listens to a dog’s vocalization and predicts emotional meaning in human language. 

 

Overview 

Understanding dog vocalizations is still a challenge for pet owners and researchers. 
 This project applies Deep Learning, Signal Processing, and Feature Engineering techniques to analyze dog barks and translate them into emotional intent (e.g., playful, alert, scared, territorial, relaxed, etc.). 

The system uses: 

 MFCC-based feature extraction. 
 A Convolutional Neural Network (CNN) classifier. 
Real-time inference through a Streamlit app. 
 Visualization models (PCA & t-SNE) to analyze features. 

 

 

 

We built a deep learning model that: 

Extracts meaningful audio features from recorded dog vocalizations 

Classifies emotions using a trained CNN model 

Provides human-readable translation via text output and probabilities 

Supports real-time microphone-based inference 

 

Probelm Statement: 

 Dogs communicate primarily through sounds such as barking, whining, growling, and whimpering. However, humans cannot accurately interpret the emotional meaning or intent behind these vocalizations. 

Proposed Solution 
We built a deep learning model that: 

Extracts meaningful audio features from recorded dog vocalizations 

Classifies emotions using a trained CNN model 

Provides human-readable translation via text output and probabilities 

Supports real-time microphone-based inference 

📁 Dataset Availability 

The dataset used in this project is proprietary and cannot be shared publicly due to licensing restrictions. 

However: 

The preprocessing pipeline is included in the repository. 

Users may use their own dataset following the same format. 

A small synthetic/sample dataset may be added later for demonstration purposes. 

 

 

 
