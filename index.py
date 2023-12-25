import re
import streamlit as st
from dataclasses import dataclass
from bot import NetraBot

title:str = "NETRA BOT"
st.title(title)

if "context" not in st.session_state:
    st.session_state.context = ""


with st.sidebar:
    st.title('NETRA CHATBOT')
    st.write('Made With (Langchain LLM) 🦜')



bot = NetraBot(
    model_id="lmsys/fastchat-t5-3b-v1.0",
    additional_parameters={
        "temperature": 2e-10,
        "max_length": 500
    },
    verbose=True
)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def generate_response(context, question):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    bot.bot_context(context=context)
    bot.bot_input(question=question)
    output = bot.bot_output()
    return output["text"]

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context_pattern = re.compile(r'context=([\s\S]*)')
            context_match = context_pattern.search(prompt)
            if context_match:
                st.session_state.context = context_match.group(1)
                response = st.session_state.context
            else:
                response = generate_response(
                    context=st.session_state.context, 
                    question=prompt
                )
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)




