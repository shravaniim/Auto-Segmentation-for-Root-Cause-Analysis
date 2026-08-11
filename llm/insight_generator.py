from llm.llm_client import client
from llm.prompt_builder import build_segment_prompt


def generate_insight(segment_info):

    prompt = build_segment_prompt(segment_info)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content