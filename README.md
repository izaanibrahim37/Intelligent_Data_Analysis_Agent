# Smart Document Analyst

An intelligent, multimodal Streamlit app that lets you upload **documents, spreadsheets, PDFs, and images** and extracts meaningful **insights, text, summaries, and visualizations** using **LLMs and OCR**.

---

## About

Smart Document Analyst is a Streamlit-based AI application that performs the following:

- **Supports multiple file formats**:  
- `.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`, and even **images**

- **Text extraction**:  
- Uses **PyMuPDF**, **docx**, **pandas**, and **Tesseract OCR** to extract readable content.

- **Automatic analysis with LLMs**:  
- Summarizes, explains, and answers your questions using the **Together AI Llama-4 model** (can be replaced with Gemini/GPT/Open-source).

- **Data visualization**:  
- Generates **histograms, box plots, bar plots, scatter plots**, and **correlation heatmaps** for numerical and categorical columns.

- **Image OCR**:  
- Detects and extracts text from image files like `.jpg`, `.png`, etc., and allows question-based interaction with it.

- **Modular backend**:
- Easy to replace LLM backend (e.g., with OpenAI/Gemini).

- **Extensible and production-ready**:
- Cleanly modularized for further extension.

---

## Folder Structure

```
Smart-Document-Analyst/
├── app.py
├── Intelligent_Data_Analysis_Agent.ipynb                   
├── README.md                
├── requirements.txt         
```

---
## How to Use Python Note book file (Intelligent_Data_Analysis_Agent.ipynb)

If you'd rather explore the logic interactively:

1. Open the file: `notebooks/SmartAnalyst.ipynb`

2. Install the required packages if not already installed:

   ```bash
   pip install streamlit pandas seaborn matplotlib pytesseract python-docx PyMuPDF
   ```

3. Run each cell to:

 - Upload a file

 - Extract content

 - Analyze and visualize manually

 - Query using the Together API

>  **Best approach to use the agent**

##  How to Use app.py file

### 1.  Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure `Tesseract OCR` is installed:
- Windows: Download from [here](https://github.com/UB-Mannheim/tesseract/wiki)
- Mac: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

---

### 2.  Set your API Key

Edit `app.py` and replace the following with your own key:
```python
TOGETHER_API_KEY = "your_actual_together_api_key_here"
```

> **Optional:** Replace with GPT, Gemini, or local Ollama models as desired.

---

### 3. ▶ Run the App

Use **Streamlit** to launch:

```bash
streamlit run app.py
```

---

### 4.  Access the App via Ngrok (for Public Access)

```bash
pip install pyngrok
ngrok config add-authtoken YOUR_AUTH_TOKEN
ngrok http 8501
```

>  **Important**: When ngrok provides links, **click on the "Visit Site" URL (the HTTPS link)** — **not localhost**, as localhost won't work across devices.

---

##  Supported File Types
--------------------------------------------------
|    File Type   |          Supported            |
|----------------|-------------------------------|
| `.pdf`         | Text extraction (PyMuPDF)     |
| `.docx`        | Text extraction (python-docx) |
| `.txt`         |                               |
| `.csv`         | Pandas + Charts               |
| `.xlsx`        | Pandas + Charts               |
| `.jpg`, `.png` | OCR + Q&A                     |
--------------------------------------------------

---

##  Features Overview

-------------------------------------------------------------------
|       Feature          |             Description                |
|------------------------|----------------------------------------|
| Text Extraction        | Extracts readable text from all files  |
| OCR from Image         | Uses Tesseract to extract image text   |
| Visualizations         | Auto-generates relevant charts         |
| LLM Analysis           | Summarization and custom Q&A           |
| Modular & Extendable   | Swap backend model with ease           |
-------------------------------------------------------------------

---

## **Author**  
**Izaan Ibrahim Sayed**  
Email: izaanahmad37@gmail.com  
GitHub: [github.com/izaanahmad37](https://github.com/izaanibrahim37) 

---

## License

MIT License – Feel free to use, modify, and share.