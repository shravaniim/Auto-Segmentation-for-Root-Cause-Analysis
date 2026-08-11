from llm.insight_generator import generate_insight

segment = {
    "Segment_Definition": "Age > 55 AND Income < 25000",
    "PSI": 0.42,
    "Delta_Gini": -0.18,
    "Delta_BR": 0.07,
    "Root_Cause_Feature": "Income"
}

result = generate_insight(segment)

print(result)