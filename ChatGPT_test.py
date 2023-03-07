# import openai
#
import openai

openai.api_key='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'
#1
messages = [
    {"role": "system", "content": "You are a kind helpful assistant."},
]

question="For the second-hand car transaction data set. The current price, mileage traveled, resale quantity, and car production year are independent variables, and the final transaction price is the dependent variable, which is fitted with a linear model. The resulting r-squared is 0.854. So, is the relationship between the dependent variable and the independent variable strong?"
# message = input("User : ")
messages.append(
    {"role": "user", "content": question},
)
chat = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=messages
)

reply = chat.choices[0].message.content
print(f"ChatGPT: {reply}")
messages.append({"role": "assistant", "content": reply})
print(messages)

#
# # 2
# import requests
#
# URL = "https://api.openai.com/v1/chat/completions"
#
# payload = {
# "model": "gpt-3.5-turbo",
# "messages": [{"role": "user", "content": f"What is the first computer in the world?"}],
# "temperature" : 1.0,
# "top_p":1.0,
# "n" : 1,
# "stream": False,
# "presence_penalty":0,
# "frequency_penalty":0,
# }
#
# headers = {
# "Content-Type": "application/json",
# "Authorization": f"Bearer {openai.api_key}"
# }
#
# response = requests.post(URL, headers=headers, json=payload, stream=False)
# response.content

