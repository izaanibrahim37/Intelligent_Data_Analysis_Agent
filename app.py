
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from together import Together
from PIL import Image
import docx
import PyPDF2
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from together import Together
import fitz
import docx
import pandas as pd
from PIL import Image
import pytesseract


def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_type == "text/plain":
        return uploaded_file.read().decode("utf-8")

    elif file_type == "text/csv":
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        return df.to_string()

    elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
        return df.to_string()

    else:
        return "Unsupported file type or cannot extract meaningful content."
        
def extract_text_from_image(image_file):
    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        return img, text
    except Exception as e:
        return None, ""

def generate_charts(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols:
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        if text_cols:
            df['text_length'] = df[text_cols[0]].apply(len)
            numeric_cols.append('text_length')
        else:
            st.write("No numeric data to visualize.")
            return

    st.subheader("Visualizations")

    for col in numeric_cols:
        st.write(f"### Histogram of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    for col in numeric_cols:
        st.write(f"### Box plot of {col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    for col in categorical_cols:
        if df[col].nunique() <= 20:
            st.write(f"### Bar plot of {col}")
            fig, ax = plt.subplots()
            counts = df[col].value_counts()
            sns.barplot(x=counts.values, y=counts.index, ax=ax)
            st.pyplot(fig)

    if len(numeric_cols) >= 2:
        st.write(f"### Scatter plot between {numeric_cols[0]} and {numeric_cols[1]}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]], ax=ax)
        st.pyplot(fig)

class DataAnalystAgent:
    def __init__(self):
        self.model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        TOGETHER_API_KEY = "tgp_v1_JLa97IreY5LiT-VlWCzmhTLOaYknDwKCkmZlk6ak194"
        self.agent = Together(api_key=TOGETHER_API_KEY)

        self.system_prompt = """
        You're a data scientist specializing in statistical analysis and visualization.
        Capabilities:
        - Process various data formats (.csv, .xlsx, .pdf, .doc, images)
        - Perform statistical analysis and data visualization
        - Answer complex data-related questions
        - Handle follow-up queries contextually
        Always respond with clear explanations and proper formatting.
        """

    def process_document(self, file_data):
        if isinstance(file_data, str):
            return pd.DataFrame({'content': [file_data]})
        elif isinstance(file_data, bytes):
            return pd.DataFrame({'content': [str(file_data)]})
        elif isinstance(file_data, pd.DataFrame):
            return file_data
        elif hasattr(file_data, 'read'):
            return pd.DataFrame({'content': [file_data.read()]})
        raise ValueError(f"Unsupported data type: {type(file_data)}")

    def analyze_text(self, text, question):
        prompt = f"""
You are a smart document analyst.

Here is the content:
{text[:2000]}

Your tasks:
- Summarize what this document is about.
- Extract useful insights.
- Answer the user's question: "{question}"
- Suggest visuals or charts if applicable.
- Do NOT mention file structure or metadata like xref, page objects, etc.
"""
        response = self.agent.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048
        )
        return response.choices[0].message.content

def show_data_and_visuals(df):
    st.subheader("Data Preview")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.write(f"Numeric columns detected: {numeric_cols}")
    if not numeric_cols:
        st.info("No numeric columns found for plotting.")
        return

    st.subheader("Visualizations")

    for col in numeric_cols:
        st.write(f"### Histogram for {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    if len(numeric_cols) >= 2:
        st.write(f"### Scatter plot between {numeric_cols[0]} and {numeric_cols[1]}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=ax)
        st.pyplot(fig)

    st.write("### Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def create_streamlit_app():
    st.title("Intelligent Data Analysis Agent")
    st.info("If you face any issues or the tool doesn't work as expected, please contact izaaanahmad37@gmail.com")


    uploaded_file = st.file_uploader("Upload your document")

    if uploaded_file is not None:
        # Check file type and process accordingly
        file_type = uploaded_file.type

        if file_type == "text/csv":
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.write(f"Numeric columns detected: {df.select_dtypes(include=['number']).columns.tolist()}")
            show_data_and_visuals(df)

        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            st.write(f"Numeric columns detected: {df.select_dtypes(include=['number']).columns.tolist()}")
            show_data_and_visuals(df)

        elif file_type == "application/pdf":
            extracted_text = extract_text_from_file(uploaded_file)
            st.success("Text extracted from PDF.")
            st.text_area("Extracted Text Preview", extracted_text[:5000], height=200)

            question = st.text_input("What do you want to know about the document?")
            if st.button("Analyze"):
                agent = DataAnalystAgent()
                result = agent.analyze_text(extracted_text, question)
                st.markdown(result)

        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = extract_text_from_file(uploaded_file)
            st.success("Text extracted from Word document.")
            st.text_area("Extracted Text Preview", extracted_text[:5000], height=200)

            question = st.text_input("What do you want to know about the document?")
            if st.button("Analyze"):
                agent = DataAnalystAgent()
                result = agent.analyze_text(extracted_text, question)
                st.markdown(result)

        elif file_type.startswith("image/"):
            img, extracted_text = extract_text_from_image(uploaded_file)
            if img:
                st.image(img, caption="Uploaded Image")
            if extracted_text.strip():
                st.success("Text extracted from Image using OCR.")
                st.text_area("Extracted Text Preview", extracted_text[:5000], height=200)
            else:
                st.info("No text detected in the image.")

            question = st.text_input("What do you want to know about the image?")
            if st.button("Analyze"):
                agent = DataAnalystAgent()
                result = agent.analyze_text(extracted_text if extracted_text.strip() else "No text extracted", question)
                st.markdown(result)

        else:
            # For other file types, fallback to text extraction
            extracted_text = extract_text_from_file(uploaded_file)
            st.success("Text extracted.")
            st.text_area("Extracted Text Preview", extracted_text[:5000], height=200)

            question = st.text_input("What do you want to know about the document?")
            if st.button("Analyze"):
                agent = DataAnalystAgent()
                result = agent.analyze_text(extracted_text, question)
                st.markdown(result)


if __name__ == "__main__":
    create_streamlit_app()
