from llm_client import client

try:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": "Hello"
            }
        ]
    )

    print("SUCCESS")
    print(response.choices[0].message.content)

except Exception as e:
    print("ERROR:")
    print(str(e))