import os
import re
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


os.environ["HUGGING_FACE_HUB_API_KEY"] = "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"

class NetraBot:
    def __init__(self, model_id: str, additional_parameters: dict, verbose: bool):
        self.is_context_set = False
        self.__llm_model = HuggingFaceHub(
            huggingfacehub_api_token=os.environ["HUGGING_FACE_HUB_API_KEY"],
            repo_id=model_id,
            model_kwargs=additional_parameters,
        )
        self.__template_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Context: {context}\nQuestion: {question}
            """
        )
        self.__chain = LLMChain(
            llm=self.__llm_model, 
            prompt=self.__template_prompt, 
            memory= ConversationBufferMemory(
                input_key="question"
            ),
            verbose=verbose
        )
    
    def bot_context(self, context: str):
        self.is_context_set = True
        self.context = context
    
    def bot_input(self, question: str):
        self.question = question
    
    def bot_output(self):
        output = self.__chain.invoke(
            {
                "context": self.context,
                "question": self.question
            }
        )
        output_result = re.sub(r'<pad>', '', output["text"])
        output["text"] = output_result
        return output
    
    
# if __name__ == "__main__":
#     bot = NetraBot(
#         model_id="lmsys/fastchat-t5-3b-v1.0",
#         additional_parameters={
#             "temperature": 2e-10,
#             "max_length": 500
#         },
#         verbose=True
#     )
#     bot.bot_context(context="""The satellite images show an adversary ship is sailing at location K, at 300 miles north west of own base which needs to be intercepted and interrogated. This may require engagement using SSMs in case of hostile action by the adversary’s ship. The mission is to be executed in the next 48 hours. The fleet must deploy suitable ship for this mission in the next 24 hours.  The speed of 25 knots (including availability of Propulsion systems and PGD systems), endurance of at least 07 days (Ration; Fuel; RO plant)""")
#     question = input("Question: ")
#     bot.bot_input(question=question)
#     output = bot.bot_output()
#     print(output)




