import os
from dotenv import load_dotenv

from tokonomics import calculate_pydantic_cost
from pydantic_ai.usage import Usage

load_dotenv()

model = os.getenv("OPENAI_LLM_MODEL")

async def compute_cost(usage: Usage) -> tuple[float, float, float]:
    
    costs = await calculate_pydantic_cost(
        model=model,
        usage=usage
    )

    if not costs:
        raise ValueError(f"Could not calculate costs for model: {model}")

    return (
        f"{costs.prompt_cost:.6f}",
        f"{costs.completion_cost:.6f}",
        f"{costs.total_cost:.6f}",
    )