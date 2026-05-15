
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

st.set_page_config(page_title="Conversational Question Rewriter", layout="centered")
st.title("💬 Conversational Question Rewriter")
st.markdown("*QReCC Project — Ask questions naturally, the system rewrites ambiguous ones using T5.*")

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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Rewriting..."):
            rewritten = rewrite_question(st.session_state.context, prompt)
            dcs = dialogue_clarity_score(prompt, rewritten, st.session_state.context)
            
            response = f"**Rewritten question:** {rewritten}\n\n**Dialogue Clarity Score (DCS):** {dcs}"
            st.markdown(response)
            
            st.session_state.context.append(prompt)
            st.session_state.context.append(rewritten)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
