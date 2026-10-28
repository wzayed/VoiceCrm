import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets
from langchain_core.tools import tool

app = FastAPI(title="FlowBridge Bilingual Voice CRM Gateway v2.2")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-3.8-live").strip()

if not MODEL_NAME.startswith("models/"):
    MODEL_NAME = f"models/{MODEL_NAME}"

GEMINI_WS_URL = (
    f"wss://generativelanguage.googleapis.com/ws/"
    f"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={GEMINI_API_KEY}"
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.2", "model": MODEL_NAME}

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
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are an executive bilingual CRM voice assistant for FlowBridge Advisory, "
                            "fluent in Arabic and English. Immediately reply out loud in the exact language spoken by the user. "
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
    print(f"\n[INFO] Browser connected (v2.2). Connecting upstream to {MODEL_NAME}...")

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

            # Send initial greeting trigger so Gemini speaks immediately upon connection
            initial_greeting = {
                "clientContent": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": "Hello! Briefly greet me in one short sentence and ask how you can help with FlowBridge CRM today."}
                            ]
                        }
                    ],
                    "turnComplete": True
                }
            }
            await gemini_ws.send(json.dumps(initial_greeting))
            print("[INFO] Sent initial greeting trigger to Gemini.")

            async def client_to_gemini():
                chunk_count = 0
                try:
                    while True:
                        message = await client_ws.receive()
                        msg_type = message.get("type")

                        if msg_type == "websocket.disconnect":
                            print(f"[INFO] Browser sent disconnect (code={message.get('code')}, chunks={chunk_count}).")
                            break

                        b64_audio = None
                        sample_rate = 16000

                        if "text" in message and message["text"]:
                            try:
                                msg = json.loads(message["text"])
                                if msg.get("type") == "text_input" and msg.get("text"):
                                    # Allow explicit text turn from client if needed
                                    await gemini_ws.send(json.dumps({
                                        "clientContent": {
                                            "turns": [{"role": "user", "parts": [{"text": msg["text"]}]}],
                                            "turnComplete": True
                                        }
                                    }))
                                    continue
                                b64_audio = msg.get("audio") or msg.get("data")
                                sample_rate = int(msg.get("sampleRate", 16000))
                            except json.JSONDecodeError:
                                continue
                        elif "bytes" in message and message["bytes"]:
                            b64_audio = base64.b64encode(message["bytes"]).decode("utf-8")

                        if b64_audio:
                            chunk_count += 1
                            if chunk_count == 1 or chunk_count % 100 == 0:
                                print(f"[INFO] Mic chunks forwarded to Gemini: {chunk_count} (rate={sample_rate})")

                            audio_msg = {
                                "realtimeInput": {
                                    "mediaChunks": [
                                        {
                                            "mimeType": f"audio/pcm;rate={sample_rate}",
                                            "data": b64_audio
                                        }
                                    ]
                                }
                            }
                            await gemini_ws.send(json.dumps(audio_msg))

                except WebSocketDisconnect as wsd:
                    print(f"[INFO] Client WebSocketDisconnect (code={wsd.code}, chunks={chunk_count}).")
                except Exception as ex:
                    print(f"[ERROR] client_to_gemini error: {type(ex).__name__}: {ex}")

            async def gemini_to_client():
                audio_out_count = 0
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
                            # Forward transcriptions if present
                            for tx_key, role in [("inputTranscription", "user"), ("outputTranscription", "assistant")]:
                                tx_obj = server_content.get(tx_key)
                                if tx_obj and tx_obj.get("text"):
                                    await client_ws.send_text(json.dumps({
                                        "type": "transcript",
                                        "role": role,
                                        "text": tx_obj["text"]
                                    }))

                            model_turn = server_content.get("modelTurn")
                            if model_turn:
                                for part in model_turn.get("parts", []):
                                    # Do not forward internal 'thought' blocks as spoken text, or label them clearly
                                    if "text" in part and part["text"]:
                                        is_thought = part.get("thought", False)
                                        await client_ws.send_text(json.dumps({
                                            "type": "transcript",
                                            "role": "thought" if is_thought else "assistant",
                                            "text": part["text"]
                                        }))
                                    inline_data = part.get("inlineData")
                                    if inline_data and "data" in inline_data:
                                        audio_out_count += 1
                                        if audio_out_count == 1 or audio_out_count % 20 == 0:
                                            print(f"[INFO] Gemini audio chunks sent to browser: {audio_out_count}")
                                        await client_ws.send_text(json.dumps({
                                            "type": "audio_delta",
                                            "audio": inline_data["data"]
                                        }))

                            if server_content.get("turnComplete"):
                                print(f"[INFO] Gemini turnComplete (total audio chunks sent: {audio_out_count}).")
                                await client_ws.send_text(json.dumps({
                                    "type": "turn_complete",
                                    "audioChunks": audio_out_count
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
                except websockets.exceptions.ConnectionClosed as cc:
                    print(f"[WARN] Gemini upstream closed connection: code={cc.code}, reason={cc.reason}")
                except Exception as ex:
                    print(f"[ERROR] gemini_to_client error: {type(ex).__name__}: {ex}")

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