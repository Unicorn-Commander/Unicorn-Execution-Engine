from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time

app = FastAPI()

pipeline = None

def set_pipeline(p):
    global pipeline
    pipeline = p

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    # For simplicity, we'll just use the last message as the prompt
    prompt = messages[-1]["content"] if messages else ""
    
    # Convert prompt to token IDs (simplified)
    # In a real implementation, you would use a proper tokenizer
    input_ids = [ord(c) for c in prompt]

    max_tokens = body.get("max_tokens", 50)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.9)

    generated_ids = pipeline.generate_tokens(
        input_ids=input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Convert token IDs back to text (simplified)
    generated_text = "".join([chr(c) for c in generated_ids])

    return JSONResponse(
        content={
            "id": f"cmpl-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gemma-3-27b-it-layer-by-layer",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(input_ids) + len(generated_ids),
            },
        }
    )

def run_server(host="0.0.0.0", port=8006):
    uvicorn.run(app, host=host, port=port)
