import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets
from langchain_core.tools import tool

app = FastAPI(title="FlowBridge Bilingual Voice CRM Gateway")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-3.8-live")

if not MODEL_NAME.startswith("models/"):
    MODEL_NAME = f"models/{MODEL_NAME}"

GEMINI_WS_URL = (
    f"wss://generativelanguage.googleapis.com/ws/"
    f"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={GEMINI_API_KEY}"
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL_NAME}

# ---------------------------------------------------------------------------
# Bilingual In-Memory CRM Database
# ---------------------------------------------------------------------------
CRM_DATABASE = [
    {
        "id": 1,
        "company": "السويدي للحلول الرقمية (Elsewedy Digital)",
        "contact": "محيي الدين عثمان (Mohie El-Din)",
        "stage": "مفاوضات متقدمة (Advanced Negotiation)",
        "value": "2,400,000 EGP",
        "owner": "كريم منصور (Karim Mansour)",
        "last_note": "طلب مراجعة اشتراطات الربط مع منظومة الفاتورة الإلكترونية والموافقة النهائية الأسبوع القادم."
    },
    {
        "id": 2,
        "company": "Nile River Logistics",
        "contact": "Tarek Abdelrahman (طارق عبد الرحمن)",
        "stage": "Proposal Sent",
        "value": "$85,000 USD",
        "owner": "David Miller",
        "last_note": "Awaiting approval on edge telematics gateways for fleet cold-chain monitoring."
    },
    {
        "id": 3,
        "company": "مجموعة الفطيم للتجارة (Al-Futtaim Trading)",
        "contact": "سارة الشريف (Sarah El-Sherif)",
        "stage": "مرحلة الترسية (Closed Won)",
        "value": "1,150,000 AED",
        "owner": "منى العبد (Mona El-Abd)",
        "last_note": "تم توقيع العقد، واجتماع إطلاق المشروع مجدول يوم الثلاثاء القادم الساعة العاشرة صباحاً."
    },
    {
        "id": 4,
        "company": "Apex Global Energy",
        "contact": "Alexander Wright",
        "stage": "Discovery Call",
        "value": "$45,000 USD",
        "owner": "Sarah Jenkins",
        "last_note": "Client wants an initial architecture overview of data governance compliance."
    }
]

# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------
@tool
def crm_search_customer(search_term: str) -> str:
    """Searches the CRM database in Arabic or English using company name or contact name."""
    term = search_term.strip().lower()
    matches = []
    for entry in CRM_DATABASE:
        searchable_text = f"{entry['company']} {entry['contact']} {entry['owner']}".lower()
        if term in searchable_text:
            matches.append(entry)

    if matches:
        return json.dumps({"found": True, "count": len(matches), "results": matches}, ensure_ascii=False)
    return json.dumps({"found": False, "message": f"No customer found matching '{search_term}'."}, ensure_ascii=False)

@tool
def crm_log_interaction_note(company_name: str, note_text: str) -> str:
    """Logs a new interaction or meeting summary note for a company in Arabic or English."""
    for entry in CRM_DATABASE:
        if company_name.strip().lower() in entry["company"].lower():
            entry["last_note"] = note_text
            return json.dumps({"status": "success", "message": f"Updated note for {entry['company']}."}, ensure_ascii=False)
    return json.dumps({"status": "error", "message": f"Company '{company_name}' not found."}, ensure_ascii=False)

crm_tools = [crm_search_customer, crm_log_interaction_note]
tools_by_name = {t.name: t for t in crm_tools}

gemini_tools = [
    {
        "functionDeclarations": [
            {
                "name": "crm_search_customer",
                "description": "Searches the CRM database in Arabic or English using company name or contact name.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "search_term": {
                            "type": "STRING",
                            "description": "Company name or contact person name to query."
                        }
                    },
                    "required": ["search_term"]
                }
            },
            {
                "name": "crm_log_interaction_note",
                "description": "Logs an interaction or meeting summary note for a company in Arabic or English.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "company_name": {
                            "type": "STRING",
                            "description": "The exact or partial company name."
                        },
                        "note_text": {
                            "type": "STRING",
                            "description": "The notes or meeting minutes to log."
                        }
                    },
                    "required": ["company_name", "note_text"]
                }
            }
        ]
    }
]

def build_setup_message() -> dict:
    return {
        "setup": {
            "model": MODEL_NAME,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": "Puck"
                        }
                    }
                }
            },
            # Enable real-time text transcripts alongside audio
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are an executive bilingual CRM voice assistant for FlowBridge Advisory, "
                            "fluent in Arabic and English. Immediately reply in the exact language spoken by the user. "
                            "Keep spoken answers concise and natural (1 to 2 sentences). "
                            "Always use your tools to look up or update CRM deals when asked."
                        )
                    }
                ]
            },
            "tools": gemini_tools
        }
    }


@app.websocket("/ws/voice-crm")
async def voice_crm_proxy(client_ws: WebSocket):
    await client_ws.accept()
    print(f"\n[INFO] Browser connected. Connecting upstream to {MODEL_NAME}...")

    if not GEMINI_API_KEY:
        await client_ws.send_text(json.dumps({
            "type": "error",
            "message": "Server configuration error: GEMINI_API_KEY is missing."
        }))
        await client_ws.close(code=1008)
        return

    try:
        async with websockets.connect(
            GEMINI_WS_URL,
            ping_interval=20,
            ping_timeout=20,
            max_size=None
        ) as gemini_ws:
            await gemini_ws.send(json.dumps(build_setup_message()))
            setup_ack = await gemini_ws.recv()
            print(f"[INFO] Handshake response: {setup_ack}")

            # Notify browser that Gemini handshake is ready
            await client_ws.send_text(json.dumps({"type": "ready"}))

            async def client_to_gemini():
                chunk_count = 0
                try:
                    while True:
                        message = await client_ws.receive()
                        if message.get("type") == "websocket.disconnect":
                            print(f"[INFO] Browser disconnected cleanly (total chunks sent: {chunk_count}).")
                            break

                        b64_audio = None
                        if "text" in message and message["text"]:
                            try:
                                msg = json.loads(message["text"])
                                # Support both {"type": "audio_input", "audio": "..."} and {"audio": "..."} or {"data": "..."}
                                b64_audio = msg.get("audio") or msg.get("data")
                            except json.JSONDecodeError:
                                continue
                        elif "bytes" in message and message["bytes"]:
                            b64_audio = base64.b64encode(message["bytes"]).decode("utf-8")

                        if b64_audio:
                            chunk_count += 1
                            if chunk_count % 100 == 0:
                                print(f"[INFO] Mic chunks forwarded to Gemini: {chunk_count}")

                            audio_msg = {
                                "realtimeInput": {
                                    "mediaChunks": [
                                        {
                                            "mimeType": "audio/pcm;rate=16000",
                                            "data": b64_audio
                                        }
                                    ]
                                }
                            }
                            await gemini_ws.send(json.dumps(audio_msg))

                except WebSocketDisconnect:
                    print(f"[INFO] Client WebSocket disconnected (total chunks sent: {chunk_count}).")
                except Exception as ex:
                    print(f"[DEBUG] client_to_gemini error: {type(ex).__name__}: {ex}")

            async def gemini_to_client():
                try:
                    async for raw_msg in gemini_ws:
                        resp = json.loads(raw_msg)

                        if "error" in resp:
                            print(f"[ERROR] Gemini upstream error: {resp['error']}")
                            await client_ws.send_text(json.dumps({
                                "type": "error",
                                "message": str(resp["error"])
                            }))
                            continue

                        server_content = resp.get("serverContent")
                        if server_content:
                            # 1. Forward User Input Transcription
                            input_tx = server_content.get("inputTranscription")
                            if input_tx and input_tx.get("text"):
                                await client_ws.send_text(json.dumps({
                                    "type": "transcript",
                                    "role": "user",
                                    "text": input_tx["text"]
                                }))

                            # 2. Forward Gemini Output Transcription
                            output_tx = server_content.get("outputTranscription")
                            if output_tx and output_tx.get("text"):
                                await client_ws.send_text(json.dumps({
                                    "type": "transcript",
                                    "role": "assistant",
                                    "text": output_tx["text"]
                                }))

                            # 3. Forward Audio Chunks & Any Text Parts
                            model_turn = server_content.get("modelTurn")
                            if model_turn:
                                for part in model_turn.get("parts", []):
                                    if "text" in part and part["text"]:
                                        await client_ws.send_text(json.dumps({
                                            "type": "transcript",
                                            "role": "assistant",
                                            "text": part["text"]
                                        }))
                                    inline_data = part.get("inlineData")
                                    if inline_data and "data" in inline_data:
                                        await client_ws.send_text(json.dumps({
                                            "type": "audio_delta",
                                            "audio": inline_data["data"]
                                        }))

                            if server_content.get("interrupted"):
                                await client_ws.send_text(json.dumps({"type": "interrupted"}))

                        tool_call = resp.get("toolCall")
                        if tool_call:
                            print(f"[INFO] Executing Tool Call: {tool_call}")
                            function_calls = tool_call.get("functionCalls", [])
                            function_responses = []

                            for call in function_calls:
                                c_id = call.get("id")
                                f_name = call.get("name")
                                f_args = call.get("args", {})

                                target_tool = tools_by_name.get(f_name)
                                if target_tool:
                                    res_str = await target_tool.ainvoke(f_args)
                                    res_obj = json.loads(res_str)
                                else:
                                    res_obj = {"error": f"Tool {f_name} not found"}

                                await client_ws.send_text(json.dumps({
                                    "type": "transcript",
                                    "role": "tool",
                                    "text": f"[CRM Tool: {f_name}] -> {json.dumps(res_obj, ensure_ascii=False)}"
                                }))

                                function_responses.append({
                                    "id": c_id,
                                    "name": f_name,
                                    "response": {"result": res_obj}
                                })

                            await gemini_ws.send(json.dumps({
                                "toolResponse": {"functionResponses": function_responses}
                            }))
                except Exception as ex:
                    print(f"[DEBUG] gemini_to_client closed: {type(ex).__name__}: {ex}")

            done, pending = await asyncio.wait(
                [asyncio.create_task(client_to_gemini()), asyncio.create_task(gemini_to_client())],
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()

    except Exception as ex:
        err_msg = f"Gemini Live Session Error: {str(ex)}"
        print(f"[ERROR] {err_msg}")
        try:
            await client_ws.send_text(json.dumps({"type": "error", "message": err_msg}))
            await client_ws.close()
        except Exception:
            pass
    finally:
        print("[INFO] Session ended.")