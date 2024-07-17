import os
from openai import OpenAI

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10810'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10810'

print(os.environ.get("OPENAI_API_KEY"));
client = OpenAI(
    # This is the default and can be omitted
    api_key='',
)

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "hi, please tell me a joke",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )

# chat_completion.choices[0].message['content']


stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")