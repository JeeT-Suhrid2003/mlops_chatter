SYSTEM_PROMPT = "You are a concise, helpful assistant."

def build_messages(user_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]