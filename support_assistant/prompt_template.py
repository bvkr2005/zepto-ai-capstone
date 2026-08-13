# =========================================================
# MODULE 3 - TASK 2
# STRUCTURED PROMPT TEMPLATE
# =========================================================

PROMPT_TEMPLATE = """
ROLE:
You are Zepto Support Assistant, an AI assistant that answers
questions about Zepto's delivery, returns, refunds, membership,
tracking, cancellations, gift cards, damaged or missing items,
and customer support policies.

CONTEXT:
Use only the Zepto policy context provided below.

{context}

TASK:
Answer the customer's question accurately using the supplied
policy context.

Customer question:
{query}

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the
provided context.
Do not invent policies, fees, timelines, benefits, contact
methods, or exceptions.

FORMAT:
Return the answer in JSON-compatible form with these fields:

answer: a concise natural-language answer
sources: a list of source document or chunk IDs used
confidence: a number from 0.0 to 1.0

LENGTH:
Keep the answer concise, preferably between 1 and 4 sentences.

FEW-SHOT EXAMPLE:

Example context:
Zepto customer support is available through in-app chat
24 hours a day, 7 days a week. Phone support is not offered.

Example customer question:
Does Zepto offer phone support?

Example answer:
{{
  "answer": "No. Zepto does not offer phone support. In-app chat is available 24/7.",
  "sources": ["doc_08"],
  "confidence": 1.0
}}

Now answer the actual customer question using only the supplied
context.
"""


# =========================================================
# BUILD PROMPT
# =========================================================

def build_prompt(context: str, query: str) -> str:
    """
    Insert retrieved policy context and the customer's
    question into the structured prompt template.
    """

    return PROMPT_TEMPLATE.format(
        context=context,
        query=query
    )


# =========================================================
# VALIDATION
# =========================================================

def validate_prompt(prompt: str) -> None:
    """
    Validate that all assignment-required prompt components
    are present.
    """

    required_sections = [
        "ROLE:",
        "CONTEXT:",
        "TASK:",
        "FORMAT:",
        "LENGTH:",
        "NEGATIVE CONSTRAINT:",
        "FEW-SHOT EXAMPLE:",
    ]

    for section in required_sections:
        if section not in prompt:
            raise ValueError(
                f"Task 2 failed: missing required section: "
                f"{section}"
            )

    # Check negative constraint explicitly.
    if "Do not invent" not in prompt:
        raise ValueError(
            "Task 2 failed: negative constraint is missing."
        )

    # Check few-shot example explicitly.
    if "Example customer question:" not in prompt:
        raise ValueError(
            "Task 2 failed: few-shot question is missing."
        )

    if "Example answer:" not in prompt:
        raise ValueError(
            "Task 2 failed: few-shot answer is missing."
        )


# =========================================================
# TEST TASK 2
# =========================================================

def main():

    print("\n========================================")
    print("MODULE 3 - SUPPORT ASSISTANT")
    print("TASK 2 - STRUCTURED PROMPT")
    print("========================================")

    example_context = (
        "Zepto gift cards are valid for 1 year "
        "from the date of issue."
    )

    example_query = (
        "How long is a Zepto gift card valid?"
    )

    prompt = build_prompt(
        context=example_context,
        query=example_query
    )

    print("\nGENERATED PROMPT")
    print("----------------------------------------")
    print(prompt)

    validate_prompt(prompt)

    print("\n========================================")
    print("TASK 2 VALIDATION")
    print("========================================")

    print("Role section: PASS")
    print("Context section: PASS")
    print("Task section: PASS")
    print("Format section: PASS")
    print("Length section: PASS")
    print("Negative constraint: PASS")
    print("Few-shot example: PASS")

    print("\n========================================")
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("========================================")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()