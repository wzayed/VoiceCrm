import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets
from langchain_core.tools import tool

app = FastAPI(title="FlowBridge Bilingual Voice CRM Gateway v2.3")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-3.8-live").strip()

# Set USE_REALTIME_TEXT=1 if the model rejects/ignores clientContent text turns
# (newer Live models only allow clientContent for seeding history).
USE_REALTIME_TEXT = os.getenv("USE_REALTIME_TEXT", "0").strip() == "1"

if not MODEL_NAME.startswith("models/"):
    MODEL_NAME = f"models/{MODEL_NAME}"

GEMINI_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={GEMINI_API_KEY}"
)

GREETING_TEXT = (
    "Hello! Briefly greet me in one short sentence and ask how you can help "
    "with FlowBridge CRM today."
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.3", "model": MODEL_NAME}


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
        "last_note": "طلب مراجعة اشتراطات الربط مع منظومة الفاتورة الإلكترونية والموافقة النهائية الأسبوع القادم.",
    },
    {
        "id": 2,
        "company": "Nile River Logistics",
        "contact": "Tarek Abdelrahman (طارق عبد الرحمن)",
        "stage": "Proposal Sent",
        "value": "$85,000 USD",
        "owner": "David Miller",
        "last_note": "Awaiting approval on edge telematics gateways for fleet cold-chain monitoring.",
    },
    {
        "id": 3,
        "company": "مجموعة الفطيم للتجارة (Al-Futtaim Trading)",
        "contact": "سارة الشريف (Sarah El-Sherif)",
        "stage": "مرحلة الترسية (Closed Won)",
        "value": "1,150,000 AED",
        "owner": "منى العبد (Mona El-Abd)",
        "last_note": "تم توقيع العقد، واجتماع إطلاق المشروع مجدول يوم الثلاثاء القادم الساعة العاشرة صباحاً.",
    },
    {
        "id": 4,
        "company": "Apex Global Energy",
        "contact": "Alexander Wright",
        "stage": "Discovery Call",
        "value": "$45,000 USD",
        "owner": "Sarah Jenkins",
        "last_note": "Client wants an initial architecture overview of data governance compliance.",
    },
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
                            "description": "Company name or contact person name to query.",
                        }
                    },
                    "required": ["search_term"],
                },
            },
            {
                "name": "crm_log_interaction_note",
                "description": "Logs an interaction or meeting summary note for a company in Arabic or English.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "company_name": {
                            "type": "STRING",
                            "description": "The exact or partial company name.",
                        },
                        "note_text": {
                            "type": "STRING",
                            "description": "The notes or meeting minutes to log.",
                        },
                    },
                    "required": ["company_name", "note_text"],
                },
            },
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
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}
                },
            },
            # Ask Gemini to send text transcripts of both sides of the audio.
            # These are siblings of generationConfig, NOT inside it.
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
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
            "tools": gemini_tools,
        }
    }


def text_turn_message(text: str) -> dict:
    if USE_REALTIME_TEXT:
        return {"realtimeInput": {"text": text}}
    return {
        "clientContent": {
            "turns": [{"role": "user", "parts": [{"text": text}]}],
            "turnComplete": True,
        }
    }


async def safe_send(client_ws: WebSocket, payload: dict) -> bool:
    try:
        await client_ws.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except Exception:
        return False


@app.websocket("/ws/voice-crm")
async def voice_crm_proxy(client_ws: WebSocket):
    await client_ws.accept()
    print(f"\n[INFO] Browser connected (v2.3). Connecting upstream to {MODEL_NAME}...")

    if not GEMINI_API_KEY:
        await safe_send(client_ws, {
            "type": "error",
            "message": "Server configuration error: GEMINI_API_KEY is missing.",
        })
        await client_ws.close(code=1008)
        return

    try:
        async with websockets.connect(
            GEMINI_WS_URL, ping_interval=20, ping_timeout=20, max_size=None
        ) as gemini_ws:
            await gemini_ws.send(json.dumps(build_setup_message()))
            setup_ack = await gemini_ws.recv()
            print(f"[INFO] Handshake response: {setup_ack}")

            # Trigger an immediate spoken greeting
            await gemini_ws.send(json.dumps(text_turn_message(GREETING_TEXT)))
            print(f"[INFO] Sent initial greeting trigger (realtime_text={USE_REALTIME_TEXT}).")

            async def client_to_gemini():
                chunk_count = 0
                try:
                    while True:
                        message = await client_ws.receive()
                        msg_type = message.get("type")

                        if msg_type == "websocket.disconnect":
                            print(f"[INFO] Browser disconnect (code={message.get('code')}, chunks={chunk_count}).")
                            break

                        b64_audio = None
                        sample_rate = 16000

                        if message.get("text"):
                            try:
                                msg = json.loads(message["text"])
                            except json.JSONDecodeError:
                                continue

                            if msg.get("type") == "text_input" and msg.get("text"):
                                await gemini_ws.send(json.dumps(text_turn_message(msg["text"])))
                                continue

                            b64_audio = msg.get("audio") or msg.get("data")
                            sample_rate = int(msg.get("sampleRate", 16000))
                        elif message.get("bytes"):
                            b64_audio = base64.b64encode(message["bytes"]).decode("utf-8")

                        if not b64_audio:
                            continue

                        chunk_count += 1
                        if chunk_count == 1 or chunk_count % 100 == 0:
                            print(f"[INFO] Mic chunks forwarded to Gemini: {chunk_count} (rate={sample_rate})")

                        # Current Live API format. The old "mediaChunks" array is deprecated
                        # and rejected/ignored by newer Live models.
                        await gemini_ws.send(json.dumps({
                            "realtimeInput": {
                                "audio": {
                                    "mimeType": f"audio/pcm;rate={sample_rate}",
                                    "data": b64_audio,
                                }
                            }
                        }))

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
                            await safe_send(client_ws, {"type": "error", "message": str(resp["error"])})
                            continue

                        server_content = resp.get("serverContent") or {}

                        # Debug: log everything except the noisy audio-chunk messages
                        is_audio_chunk = bool(server_content.get("modelTurn")) and not server_content.get("turnComplete")
                        if not is_audio_chunk:
                            print(f"[UPSTREAM] keys={list(resp.keys())} serverContent_keys={list(server_content.keys())}")

                        if server_content:
                            for tx_key, role in [("inputTranscription", "user"), ("outputTranscription", "assistant")]:
                                tx_obj = server_content.get(tx_key)
                                if tx_obj and tx_obj.get("text"):
                                    await safe_send(client_ws, {
                                        "type": "transcript",
                                        "role": role,
                                        "text": tx_obj["text"],
                                    })

                            model_turn = server_content.get("modelTurn")
                            if model_turn:
                                for part in model_turn.get("parts", []):
                                    if part.get("text"):
                                        await safe_send(client_ws, {
                                            "type": "transcript",
                                            "role": "thought" if part.get("thought") else "assistant",
                                            "text": part["text"],
                                        })
                                    inline_data = part.get("inlineData")
                                    if inline_data and "data" in inline_data:
                                        audio_out_count += 1
                                        if audio_out_count == 1 or audio_out_count % 20 == 0:
                                            print(f"[INFO] Gemini audio chunks sent to browser: {audio_out_count}")
                                        await safe_send(client_ws, {
                                            "type": "audio_delta",
                                            "audio": inline_data["data"],
                                        })

                            if server_content.get("turnComplete"):
                                print(f"[INFO] Gemini turnComplete (audio chunks this session: {audio_out_count}).")
                                await safe_send(client_ws, {
                                    "type": "turn_complete",
                                    "audioChunks": audio_out_count,
                                })

                            if server_content.get("interrupted"):
                                await safe_send(client_ws, {"type": "interrupted"})

                        tool_call = resp.get("toolCall")
                        if tool_call:
                            print(f"[INFO] Executing Tool Call: {tool_call}")
                            function_responses = []

                            for call in tool_call.get("functionCalls", []):
                                c_id = call.get("id")
                                f_name = call.get("name")
                                f_args = call.get("args", {})

                                target_tool = tools_by_name.get(f_name)
                                if target_tool:
                                    try:
                                        res_str = await target_tool.ainvoke(f_args)
                                        res_obj = json.loads(res_str)
                                    except Exception as tex:
                                        res_obj = {"error": f"Tool {f_name} failed: {tex}"}
                                else:
                                    res_obj = {"error": f"Tool {f_name} not found"}

                                await safe_send(client_ws, {
                                    "type": "transcript",
                                    "role": "tool",
                                    "text": f"[CRM Tool: {f_name}] -> {json.dumps(res_obj, ensure_ascii=False)}",
                                })

                                function_responses.append({
                                    "id": c_id,
                                    "name": f_name,
                                    "response": {"result": res_obj},
                                })

                            await gemini_ws.send(json.dumps({
                                "toolResponse": {"functionResponses": function_responses}
                            }))

                except websockets.exceptions.ConnectionClosed as cc:
                    print(f"[WARN] Gemini upstream closed: code={cc.code}, reason={cc.reason}")
                    await safe_send(client_ws, {
                        "type": "error",
                        "message": f"Gemini closed the session (code {cc.code}): {cc.reason}",
                    })
                except Exception as ex:
                    print(f"[ERROR] gemini_to_client error: {type(ex).__name__}: {ex}")
                    await safe_send(client_ws, {
                        "type": "error",
                        "message": f"Gateway error: {type(ex).__name__}: {ex}",
                    })

            done, pending = await asyncio.wait(
                [asyncio.create_task(client_to_gemini()), asyncio.create_task(gemini_to_client())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except Exception as ex:
        err_msg = f"Gemini Live Session Error: {str(ex)}"
        print(f"[ERROR] {err_msg}")
        await safe_send(client_ws, {"type": "error", "message": err_msg})
        try:
            await client_ws.close()
        except Exception:
            pass
    finally:
        print("[INFO] Session ended.")
