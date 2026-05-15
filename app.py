
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

st.set_page_config(page_title="Conversational Question Rewriter", layout="centered")

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = load_model()

def rewrite_question(context, question):
    context_str = " ||| ".join(context[-3:])
    input_text = f"rewrite question: {question} context: {context_str}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def dialogue_clarity_score(original, rewritten, context):
    orig_words = set(str(original).lower().split())
    rew_words = set(str(rewritten).lower().split())
    context_words = set(" ".join(context).lower().split())
    ambig = {"it","they","he","she","this","that","these","those","there"}
    stopwords = {"the","a","an","is","are","was","were","do","does","did","of","in","on","at"}
    orig_ambig = len(orig_words & ambig)
    rew_ambig = len(rew_words & ambig)
    ambiguity_reduction = max(0, (orig_ambig - rew_ambig) / (orig_ambig + 1))
    orig_keywords = orig_words - stopwords - ambig
    preservation = len(orig_keywords & rew_words) / (len(orig_keywords) + 1)
    new_words = rew_words - orig_words
    enrichment = len(new_words & context_words) / (len(new_words) + 1)
    return round(0.4 * ambiguity_reduction + 0.3 * preservation + 0.3 * enrichment, 4)

st.title("Conversational Question Rewriter")
st.markdown("**QReCC Project** — Rewrite ambiguous questions using conversation context and T5.")

st.subheader("Conversation Context")
context_input = st.text_area("Enter each turn separated by |||", placeholder="Turn 1 ||| Turn 2 ||| Turn 3")

st.subheader("Ambiguous Question")
question_input = st.text_input("Enter the ambiguous question", placeholder="e.g. What are its effects?")

if st.button("Rewrite Question"):
    if question_input:
        context = [c.strip() for c in context_input.split("|||") if c.strip()]
        with st.spinner("Rewriting..."):
            rewritten = rewrite_question(context, question_input)
            dcs = dialogue_clarity_score(question_input, rewritten, context)
        st.success("Done!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rewritten Question", rewritten)
        with col2:
            st.metric("Dialogue Clarity Score (DCS)", dcs)
    else:
        st.warning("Please enter a question.")
