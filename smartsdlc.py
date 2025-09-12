import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader

# ------------------ Load Model ------------------
model_name = "ibm-granite/granite-3.3-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------ Core Functions ------------------
def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def eco_tips_generator(problem_keywords):
    prompt = (
        f"Generate practical and actionable eco-friendly tips for sustainable living "
        f"related to: {problem_keywords}. Provide specific solutions and suggestions."
    )
    return generate_response(prompt, max_length=1000)

def policy_summarization(pdf_file, policy_text):
    if pdf_file is not None:
        content = extract_text_from_pdf(pdf_file)
        summary_prompt = (
            f"Summarize the following policy document and extract the most important points, "
            f"key provisions, and implications:\n\n{content}"
        )
    else:
        summary_prompt = (
            f"Summarize the following policy document and extract the most important points, "
            f"key provisions, and implications:\n\n{policy_text}"
        )
    return generate_response(summary_prompt, max_length=1200)

def carbon_footprint_estimator(activity_details):
    prompt = (
        f"Estimate the carbon footprint (in kg CO₂ per month) based on the following lifestyle details: "
        f"{activity_details}. Then, suggest ways to reduce the footprint effectively."
    )
    return generate_response(prompt, max_length=800)

def green_tech_ideas(sector):
    prompt = (
        f"Suggest innovative eco-friendly and sustainable technology ideas for the sector: {sector}. "
        f"Include practical applications, scalability, and environmental impact."
    )
    return generate_response(prompt, max_length=900)

# ------------------ Gradio UI ------------------
with gr.Blocks() as app:
    gr.Markdown("# 𝑹𝑬𝑫 𝑫𝒀𝑵𝑶 𝑨𝑰")

    with gr.Tabs():
        # Tab 1: Eco Tips
        with gr.TabItem("𝑮𝒓𝒆𝒆𝒏 𝑻𝒓𝒂𝒗𝒆𝒍 𝑪𝒉𝒐𝒊𝒄𝒆𝒔"):
            with gr.Row():
                with gr.Column():
                    keywords_input = gr.Textbox(
                        label="Environmental Problem/Keywords",
                        placeholder="e.g., plastic, solar, water waste, energy saving...",
                        lines=3
                    )
                    generate_tips_btn = gr.Button("Generate Eco Tips")
                with gr.Column():
                    tips_output = gr.Textbox(label="Sustainable Living Tips", lines=10, elem_id="tips_box")
                    tips_copy = gr.HTML("""
                        <button onclick="navigator.clipboard.writeText(
                            document.querySelector('#tips_box textarea').value)"> Copy</button>
                    """)
            generate_tips_btn.click(
                eco_tips_generator,
                inputs=keywords_input,
                outputs=tips_output
            )

        # Tab 2: Policy Summarization
        with gr.TabItem("𝑩𝒊𝒐𝒅𝒊𝒗𝒆𝒓𝒔𝒊𝒕𝒚 𝑷𝒓𝒐𝒕𝒆𝒄𝒕𝒊𝒐𝒏 𝑨𝒄𝒕𝒔"):
            with gr.Row():
                with gr.Column():
                    pdf_upload = gr.File(label="Upload Policy PDF", file_types=[".pdf"])
                    policy_text_input = gr.Textbox(
                        label="Or paste policy text here",
                        placeholder="Paste policy document text...",
                        lines=5
                    )
                    summarize_btn = gr.Button("Summarize Policy")
                with gr.Column():
                    summary_output = gr.Textbox(label="Policy Summary & Key Points", lines=20, elem_id="summary_box")
                    summary_copy = gr.HTML("""
                        <button onclick="navigator.clipboard.writeText(
                            document.querySelector('#summary_box textarea').value)"> Copy</button>
                    """)
            summarize_btn.click(
                policy_summarization,
                inputs=[pdf_upload, policy_text_input],
                outputs=summary_output
            )

        # Tab 3: Carbon Footprint Estimator
        with gr.TabItem("𝑬𝒄𝒐-𝑭𝒓𝒊𝒆𝒏𝒅𝒍𝒚 𝑯𝒂𝒃𝒊𝒕𝒔"):
            with gr.Row():
                with gr.Column():
                    activity_input = gr.Textbox(
                        label="Enter your daily/weekly activities",
                        placeholder="e.g., I drive 20km daily, use AC 8 hours/day, eat meat 3 times/week...",
                        lines=5
                    )
                    footprint_btn = gr.Button("Estimate Carbon Footprint")
                with gr.Column():
                    footprint_output = gr.Textbox(label="Carbon Footprint & Suggestions", lines=10, elem_id="footprint_box")
                    footprint_copy = gr.HTML("""
                        <button onclick="navigator.clipboard.writeText(
                            document.querySelector('#footprint_box textarea').value)"> Copy</button>
                    """)
            footprint_btn.click(
                carbon_footprint_estimator,
                inputs=activity_input,
                outputs=footprint_output
            )

        # Tab 4: Green Technology Ideas
        with gr.TabItem("𝑬𝒄𝒐-𝑭𝒓𝒊𝒆𝒏𝒅𝒍𝒚 𝑴𝒂𝒕𝒆𝒓𝒊𝒂𝒍𝒔"):
            with gr.Row():
                with gr.Column():
                    sector_input = gr.Textbox(
                        label="Enter a sector/industry",
                        placeholder="e.g., agriculture, transportation, fashion, construction...",
                        lines=2
                    )
                    ideas_btn = gr.Button("Generate Green Tech Ideas")
                with gr.Column():
                    ideas_output = gr.Textbox(label="Eco-Friendly Innovation Ideas", lines=10, elem_id="ideas_box")
                    ideas_copy = gr.HTML("""
                        <button onclick="navigator.clipboard.writeText(
                            document.querySelector('#ideas_box textarea').value)"> Copy</button>
                    """)
            ideas_btn.click(
                green_tech_ideas,
                inputs=sector_input,
                outputs=ideas_output
            )

# Launch the app
app.launch(share=True)