import requests
import json

# --- Monkey patch LLMInterface ---
from oraska.agents.llm_interface import LLMInterface

async def generate_patch(self, prompt, **kwargs):
    # 简单返回模拟计划，Oraska 执行时不会报错
    return {
        "plan": f"[Simulated plan for prompt]: {prompt}",
        "choices": [{"text": prompt}]
    }

LLMInterface.generate = generate_patch
# --- End patch ---

API_URL = "http://localhost:8000/tasks/execute"

def send_task(description: str, task_id: str = None):
    payload = {
        "id": task_id,
        "description": description,
        "context": {}
    }
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"{response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=== Oraska Interactive Task Console ===")
    print("Type 'exit' to quit.\n")

    counter = 1
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        task_id = f"interactive_{counter}"
        result = send_task(user_input, task_id)
        print("Oraska:", json.dumps(result, indent=4))
        counter += 1

if __name__ == "__main__":
    main()
