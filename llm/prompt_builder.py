def build_segment_prompt(segment_info: dict) -> str:

    return f"""
You are a senior credit risk model validator.

Analyze the following degraded segment.

Segment:
{segment_info.get('Segment_Definition')}

PSI:
{segment_info.get('PSI')}

Gini Drop:
{segment_info.get('Delta_Gini')}

Bad Rate Shift:
{segment_info.get('Delta_BR')}

Root Cause Feature:
{segment_info.get('Root_Cause_Feature')}

Generate:

1. Executive Summary
2. Business Impact
3. Possible Root Cause
4. Recommended Action

Keep response concise.
"""