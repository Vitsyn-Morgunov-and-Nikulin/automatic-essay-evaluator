import streamlit as st
import seaborn as sns
from demo.utils import load_model, process_text
import re

st.set_page_config(
    page_title="BERT Keyword Extractor",
    page_icon="🎈",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


st.header("🔑 Automated Essay Evaluator")

with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """
        -   This application demonstrates how automated essay evaluation works: given as an input text with max. \
        length of 512, it scores it (from 1.0 to 4.0) for different criteria: cohesion, syntax, vocabulary, \
        phraseology, grammar and conventions.
        -   This solution is based on fine-tuned deberta-v3-large model.
        """
    )

    st.markdown("")

st.markdown("")
st.markdown("## 📌 **Paste document**", unsafe_allow_html=True)
with st.form(key="my_form"):
    _, c2, _ = st.columns([0.07, 5, 0.07])

    with c2:
        doc = st.text_area(
            "Paste your text below (max 500 words)",
            height=510,
        )

        MAX_WORDS = 500

        res = len(re.findall(r"\w+", doc))
        doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Get me the data!")

if not submit_button:
    st.stop()

st.markdown("## 🎈  **Check results**")

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

st.header("")

model = load_model()
df = process_text(doc, model)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Grade",
    ],
)


format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

st.table(df)
