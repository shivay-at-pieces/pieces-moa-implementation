import asyncio
import os
from application import client, connect_api, api_client
import requests


if client.WellKnownApi(api_client).get_well_known_health() != "ok":
    raise ConnectionError("Please open the pieces server")

extensions = [e.value for e in client.ClassificationSpecificEnum]
opensource_application = connect_api()
models_api = client.ModelsApi(api_client)

# Get models from Client
api_response = models_api.models_snapshot()
models = {model.name: model.id for model in api_response.iterable if model.cloud or model.downloaded}

# Set default model from Client
default_model_name = "GPT-4 Chat Model"
model_id = models[default_model_name]
models_name = list(models.keys())
default_model_index = models_name.index(default_model_name)

user_prompt = "What are 3 fun things to do in SF?"
reference_model_ids = [
    "9c909eeb-03ed-4dd0-a4f9-2bd5dbe98674", # Granite 8B
    "15b3c0d6-ac0a-474e-a4ce-a355bab4bfc5", # Gemma 7B
    "10ccd298-9d97-46fa-9c27-d0bc851a599f", # Phi-3 4k 
    "95014296-b1b2-4a47-8eea-14b121baa20a", # Mistral 7B
]

aggregator_model = "f674e8ad-8a2c-4c38-a8a7-45a481fb1676" # LLama3
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

async def run_llm(model_id):
    question = client.QGPTQuestionInput(
        query=user_prompt,
        relevant={"iterable": []},
        model=model_id
    )

    try:
        # Create an Instance of Question Output 
        question_output = client.QGPTApi(api_client).question(question)

        # Getting the answer
        answers = question_output.answers.iterable[0].text
        return answers
    except requests.exceptions.JSONDecodeError:
        return "Failed to decode JSON response"
    except Exception as e:
        return f"An error occurred: {str(e)}"

async def main():
    results = await asyncio.gather(*[run_llm(model_id) for model_id in reference_model_ids])
    
    # Print individual results
    for i, result in enumerate(results):
        print(f"Result from model {i+1}: {result}\n")

    # Combine the results into a single string for the aggregator prompt
    combined_responses = "\n".join([f"{i+1}. {response}" for i, response in enumerate(results)])
    final_prompt = f"{aggregator_system_prompt}\n{combined_responses}"

    question = client.QGPTQuestionInput(
        query=final_prompt,
        relevant={"iterable": []},
        model=aggregator_model
    )

    try:
        # Create an Instance of Question Output 
        question_output = client.QGPTApi(api_client).question(question)

        # Getting the answer
        final_answer = question_output.answers.iterable[0].text
        print("\nFinal Aggregated Answer:\n")
        print(final_answer)
    except requests.exceptions.JSONDecodeError:
        print("Failed to decode JSON response")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

asyncio.run(main())