import pandas as pd
import gradio as gr
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def read_file(file):
    """Read a file and return a DataFrame based on its extension."""
    file_extension = file.name.split('.')[-1]
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file.name, encoding='utf-8')
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file.name)
        else:
            raise ValueError("Unsupported file format")
    except UnicodeDecodeError:
        df = pd.read_csv(file.name, encoding='ISO-8859-1')

    return df

def clean_data(df):
    """Clean the DataFrame by handling missing values and encoding categorical variables."""
    missing_before = df.isnull().sum()
    
    df.fillna(df.median(numeric_only=True), inplace=True)
    missing_after = df.isnull().sum()

    cleaning_report = {
        "missing_before": missing_before,
        "missing_after": missing_after,
    }
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                if df[col].nunique() < 10:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                else:
                    df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df, cleaning_report

def visualize_data(df, target_variable):
    """Generate visualizations for the DataFrame and save to files."""
    heatmap_path = "correlation_heatmap.png"
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(heatmap_path)
    plt.close()

    hexbin_plot_paths = []
    for col in df.columns:
        if col != target_variable:
            hexbin_plot_path = f"hexbin_{col}.png"
            plt.figure(figsize=(10, 6))
            plt.hexbin(df[col], df[target_variable], gridsize=50, cmap='Blues')
            plt.colorbar(label='Count in Bin')
            plt.title(f'Hexbin Plot of {col} vs {target_variable}')
            plt.xlabel(col)
            plt.ylabel(target_variable)
            plt.savefig(hexbin_plot_path)
            plt.close()
            hexbin_plot_paths.append(hexbin_plot_path)

    return heatmap_path, hexbin_plot_paths

def process_file(file, target_variable):
    """Process the uploaded file and visualize the results."""
    try:
        df = read_file(file)
        head = df.head()
        df, cleaning_report = clean_data(df)
        heatmap_path, hexbin_plots = visualize_data(df, target_variable)
        
        # Prepare output message
        report = f"Visualization complete. Check the plots.\n\n"
        report += "Data Head:\n" + str(head) + "\n\n"
        report += "Missing Values Report:\n" + str(cleaning_report['missing_before']) + "\n"
        report += "Missing Values After Cleaning:\n" + str(cleaning_report['missing_after']) + "\n"

        return report, heatmap_path, hexbin_plots
    except Exception as e:
        return str(e), None, []

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Gradio Interface for Data Processing
data_interface = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(label="Upload CSV or Excel File"),
        gr.Textbox(label="Target Variable Name")
    ],
    outputs=[
        "text",  # Message about completion
        gr.Image(type="filepath"),  # Correlation heatmap
        gr.Gallery(label="Hexbin Plots")  # Hexbin plots
    ],
    title="Data Visualization Tool",
    description="Upload a CSV or Excel file to visualize relationships in your dataset."
)

# Gradio Interface for Chatbot
chat_interface = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

# Launch both interfaces
if __name__ == "__main__":
    data_interface.launch()
    chat_interface.launch()
