import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the tokenizer and model outside the app to avoid reloading on each input
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    return nlp

# Streamlit app
def main():
    st.title("Named Entity Recognition App")
    st.write("Enter your text below to identify entities like person names, organizations, and locations:")

    # Text input from the user
    user_input = st.text_area("Enter text", placeholder="Type your text here...", height=150)

    # Submit button
    if st.button("Submit"):
        # Load the NER pipeline model
        nlp = load_model()

        if user_input:
            st.write("### Extracted Entities:")
            # Process the user input and get NER results
            ner_results = nlp(user_input)

            # Display the entities in a readable format
            if ner_results:
                for result in ner_results:
                    word = result['word']
                    entity = result['entity']
                    st.write(f"**Word:** {word}, **Entity:** {entity}")
            else:
                st.write("No entities found.")

if __name__ == "__main__":
    main()
