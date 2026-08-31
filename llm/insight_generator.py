from llm.llm_client import client, MODEL_DEPLOYMENT
from llm.prompt_builder import build_segment_prompt, build_qa_prompt


def generate_insight(segment_info):

    prompt = build_segment_prompt(segment_info)

    response = client.chat.completions.create(
        model=MODEL_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def answer_data_question(question: str, context_text: str) -> str:
    """Free-form Q&A over whatever run results are currently on screen.
    Same client/call shape as generate_insight -- just a different prompt
    (a question + a data context, instead of one fixed segment template)."""

    prompt = build_qa_prompt(question, context_text)

    response = client.chat.completions.create(
        model=MODEL_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content