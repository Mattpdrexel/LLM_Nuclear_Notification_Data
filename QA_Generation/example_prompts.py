

# Prompt templates for multi-step pipeline
COMPONENT_ID_PROMPT_TEMPLATE = """
# ─── Background ─────────────────────────────────────────
{CW_TRAINING}

Here are several notification entries:

{snippet}

Identify the single primary component these notifications relate to.
Your answer must:
- Include a specific equipment identifier and descriptor if appropriate (e.g., "11A CWP", "12A Travel Screen", "13B Screenwash Pump").
- Alternatively, can include class of component (e.g., "CWP", "Travel Screen", "Screenwash Pump").
- Not be a generic term like "pump", "lube oil", "motor", etc.

Return only the exact component string.
"""

QUESTION_GEN_PROMPT_TEMPLATE = """
# ─── Background ─────────────────────────────────────────
{background}

# ─── Examples ───────────────────────────────────────────
Example 1:
Q: {seed1_question}
A: {seed1_answer}

Example 2:
Q: {seed2_question}
A: {seed2_answer}

# ─── Context Notifications ─────────────────────────────
{component_context}

# ─── Your Task ─────────────────────────────────────────
Using the notifications above and focusing on component "{component}", craft one single-sentence question that is both insightful and grounded. Return only the question.
"""

COMBINED_QA_PROMPT_TEMPLATE = """
# ─── Background ─────────────────────────────────────────
{background}

# ─── Examples ───────────────────────────────────────────
Example 1:
Q: {seed1_question}
A: {seed1_answer}

Example 2:
Q: {seed2_question}
A: {seed2_answer}

# ─── Context Notifications ─────────────────────────────
{question_context}

# ─── Draft Question ─────────────────────────────────────
{draft_question}

# ─── Your Task ─────────────────────────────────────────
Using ONLY the notifications above (all of which mention {component}), refine the draft question if needed to be highly insightful and grounded, then provide a concise answer.
The answer MUST be directly supported from the reference notifications. The references MUST be directly related to the component "{component}" (for example, if issue is related to 11A CWP, don't reference notifications related to 12A CWP).
Return a JSON object with keys "question", "answer", and "references" where:
  - "references" is the list of notifications cited (e.g., "Not. Notification – ShortText (YYYY-MM-DD)").
  - If the provided notifications do not contain sufficient information to directly answer the question, set "answer" to "Insufficient information to answer this question." and "references" to [].
We will use these Q&A pairs to fine-tune a later model, so please return strictly valid JSON.
"""
