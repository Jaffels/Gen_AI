# Gen_AI
Group project for Generative AI
Johan Ferreira, Nhat Bui, Thilo Holstein, Wenxing Xu 

## Required Packages
streamlit
PyPDF2
langchain
langchain-openai
langchain-community
faiss-cpu
openai
python-dotenv
numpy
#### Run this code in your virtual enviroment's terminal
pip install streamlit PyPDF2 langchain langchain-openai langchain-community faiss-cpu openai python-dotenv numpy

#### For improved performance on Mac install
xcode-select --install  
pip install watchdog

## Bash run the chatbot (run in your terminal)
streamlit run chatbot.py

## Break down of data collection strategies and Universities
### Gemini sourced
- FHGR
- OST
- EPFL
- SUPSI
- University of Zurich
- University of Western Switzerland
- Bern
- UniNE

### Manually sourced
- BFH
- FHNW
- HSLU
- UGE

### Webscrapping sourced
- ETH
- ZHAW

Questions
- What is the cost per semester?
- Do the University have a Masters program in Data Science?
- Does the university have a world wide or QS ranking?