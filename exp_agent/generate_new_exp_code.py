"""
action_itemの記述に沿って元コードを改修した新しいコードを生成する
"""

import argparse
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

system_template="あなたはデータサイエンティストです。元コードと改善点に基づいて修正したコードを出力してください。返答に説明は不要です。そのまま実行可能なpythonコードの中身のみを返答してください。また、適宜コメントを日本語で追加してください。"
human_template="""
元コード:
{code}

改善点:
{action_item}"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_code_path", required=True)
parser.add_argument("--action_item", required=True)
parser.add_argument("--llm_model", default="text-davinci-003", required=True)
args = parser.parse_args() 

exp_code_path = args.exp_code_path
action_item = args.action_item
llm_model = args.llm_model

with open(exp_code_path, 'r') as file:
    org_code = file.read()

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
prompt_message_list = chat_prompt.format_prompt(code=org_code, action_item=action_item).to_messages()

print(prompt_message_list)
chat = ChatOpenAI(model_name=llm_model)
with get_openai_callback() as cb:
    new_code = chat(prompt_message_list)
    print(cb)
print(new_code.content)

with open(exp_code_path, "w") as file:
   file.write(new_code.content)