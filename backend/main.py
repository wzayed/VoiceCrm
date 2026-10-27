import os
import json
import base64
import asyncio

import websockets
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from langchain_core.tools import tool

app = FastAPI(title="Bilingual Voice CRM Gateway")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-3.8-live")

# Ensure MODEL_NAME always starts with "models/"
if not MODEL_NAME.startswith("models/"):
    MODEL_NAME = f"models/{MODEL_NAME}"

GEMINI_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={GEMINI_API_KEY}"
)

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

# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------
@tool
def crm_search_customer(search_term: str) -> str:
    """Searches the CRM database in Arabic or English using company name, contact name, stage, value, owner, or last_note."""
    term = search_term.strip().lower()
    matches = []
    for entry in CRM_DATABASE:
        searchable_text = f"{entry['company']} {entry['contact']} {entry['owner']} {entry['value']} {entry['stage']} {entry['last_note']}".lower()
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
                "description": "Searches the CRM database in Arabic or English using company name, contact name, stage, value, owner, or last_note.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "search_term": {
                            "type": "STRING",
                            "description": "company name, contact name, stage, value, owner, or last_note.",
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
                # The Live API accepts ONE response modality per session.
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}
                },
            },
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are an executive bilingual CRM voice assistant fluent in Arabic and English. "
                            "When the user speaks to you, immediately reply out loud in that same language. "
                            "Keep your spoken answers brief and natural (1 to 2 sentences). "
                            "Always use your tools to query or update the CRM database when asked."
                        )
                    }
                ]
            },
            "tools": gemini_tools,
            # Text transcripts of what Gemini HEARD and what it SAID (diagnostics + UI)
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
    }


async def run_tool(name: str, args: dict) -> dict:
    """Execute a CRM tool and return a JSON-object result for Gemini."""
    tool_obj = tools_by_name.get(name)
    if tool_obj is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        # LangChain tools are sync; keep the event loop free.
        raw = await asyncio.to_thread(tool_obj.invoke, args or {})
        try:
            return {"result": json.loads(raw)}
        except (TypeError, json.JSONDecodeError):
            return {"result": raw}
    except Exception as ex:  # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}"}


def audio_chunk_message(b64_audio: str) -> str:
    return json.dumps(
        {"realtimeInput": {"audio": {"mimeType": "audio/pcm;rate=16000", "data": b64_audio}}}
    )


async def safe_send_client(client_ws: WebSocket, payload: dict) -> None:
    try:
        await client_ws.send_text(json.dumps(payload, ensure_ascii=False))
    except Exception:  # client already gone
        pass


@app.websocket("/ws/voice-crm")
async def voice_crm_proxy(client_ws: WebSocket):
    await client_ws.accept()
    print(f"\n[INFO] Browser connected. Attempting Gemini connection with model: {MODEL_NAME}")

    if not GEMINI_API_KEY:
        await safe_send_client(
            client_ws,
            {"type": "error", "message": "Server configuration error: GEMINI_API_KEY is missing."},
        )
        await client_ws.close(code=1008)
        return

    try:
        async with websockets.connect(GEMINI_WS_URL, max_size=None) as gemini_ws:
            print("[INFO] Connected to Google Gemini upstream successfully.")
            await gemini_ws.send(json.dumps(build_setup_message()))

            setup_ack = await gemini_ws.recv()
            print(f"[INFO] Handshake response: {setup_ack!r}")
            try:
                ack = json.loads(setup_ack)
            except json.JSONDecodeError:
                ack = {}
            if "setupComplete" not in ack:
                raise RuntimeError(f"Gemini setup failed: {setup_ack!r}")

            # TEST MODE: make Gemini speak without using the mic.
            # If you hear this greeting, Gemini -> backend -> browser audio works,
            # and the problem is in the mic -> Gemini direction.
            if os.getenv("TEST_GREETING") == "1":
                print("[INFO] TEST_GREETING enabled: asking Gemini to say hello.")
                await gemini_ws.send(
                    json.dumps(
                        {
                            "clientContent": {
                                "turns": [
                                    {
                                        "role": "user",
                                        "parts": [{"text": "Say hello in one short sentence."}],
                                    }
                                ],
                                "turnComplete": True,
                            }
                        }
                    )
                )

            stats = {"audio_in": 0, "audio_out": 0}

            async def client_to_gemini():
                """Browser -> Gemini. Exits cleanly when the browser disconnects."""
                try:
                    while True:
                        message = await client_ws.receive()

                        # THE FIX: stop reading once a disconnect arrives.
                        if message["type"] == "websocket.disconnect":
                            print(f"[INFO] Browser disconnected (code={message.get('code')}).")
                            break

                        data = message.get("bytes")
                        text = message.get("text")

                        if data is not None:
                            b64_audio = base64.b64encode(data).decode("utf-8")
                            await gemini_ws.send(audio_chunk_message(b64_audio))
                            stats["audio_in"] += 1
                        elif text is not None:
                            try:
                                msg = json.loads(text)
                            except json.JSONDecodeError:
                                continue
                            if msg.get("type") == "audio_input" and msg.get("audio"):
                                await gemini_ws.send(audio_chunk_message(msg["audio"]))
                                stats["audio_in"] += 1
                                if stats["audio_in"] in (1, 50, 500):
                                    print(f"[INFO] Mic chunks forwarded to Gemini: {stats['audio_in']}")
                except WebSocketDisconnect:
                    pass
                except websockets.exceptions.ConnectionClosed:
                    pass

            async def gemini_to_client():
                """Gemini -> Browser. Also executes tool calls."""
                try:
                    async for raw_msg in gemini_ws:
                        resp = json.loads(raw_msg)

                        # Log anything that is not plain audio so we can see errors/goAway/etc.
                        sc = resp.get("serverContent") or {}
                        if not sc.get("modelTurn"):
                            print(f"[DEBUG] Gemini message: {json.dumps(resp, ensure_ascii=False)[:400]}")

                        server_content = resp.get("serverContent")
                        if server_content:
                            for key, role in (("inputTranscription", "user"), ("outputTranscription", "assistant")):
                                tr = server_content.get(key)
                                if tr and tr.get("text"):
                                    await safe_send_client(
                                        client_ws, {"type": "transcript", "role": role, "text": tr["text"]}
                                    )

                            if server_content.get("interrupted"):
                                await safe_send_client(client_ws, {"type": "interrupted"})

                            model_turn = server_content.get("modelTurn")
                            if model_turn:
                                for part in model_turn.get("parts", []):
                                    inline_data = part.get("inlineData")
                                    if inline_data and "data" in inline_data:
                                        stats["audio_out"] += 1
                                        if stats["audio_out"] == 1:
                                            print("[INFO] First audio chunk from Gemini received.")
                                        await safe_send_client(
                                            client_ws,
                                            {"type": "audio_delta", "audio": inline_data["data"]},
                                        )

                            if server_content.get("turnComplete"):
                                await safe_send_client(client_ws, {"type": "turn_complete"})

                        tool_call = resp.get("toolCall")
                        if tool_call:
                            responses = []
                            for fc in tool_call.get("functionCalls", []):
                                name = fc.get("name")
                                print(f"[INFO] Tool call: {name} {fc.get('args')}")
                                result = await run_tool(name, fc.get("args") or {})
                                responses.append(
                                    {"id": fc.get("id"), "name": name, "response": result}
                                )
                            await gemini_ws.send(
                                json.dumps({"toolResponse": {"functionResponses": responses}})
                            )
                except websockets.exceptions.ConnectionClosed as ex:
                    msg = f"Gemini closed the connection: {ex}"
                    print(f"[ERROR] {msg}")
                    await safe_send_client(client_ws, {"type": "error", "message": msg})

            tasks = [
                asyncio.create_task(client_to_gemini()),
                asyncio.create_task(gemini_to_client()),
            ]
            # When EITHER direction ends, tear down the other one.
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for t in done:
                if not t.cancelled() and t.exception():
                    ex = t.exception()
                    print(f"[ERROR] stream task failed: {type(ex).__name__}: {ex}")

    except websockets.exceptions.InvalidHandshake as ex:
        status = getattr(ex, "status_code", None) or getattr(
            getattr(ex, "response", None), "status_code", "unknown"
        )
        err_msg = (
            f"Gemini rejected connection (HTTP {status}). "
            "Your API key or model tier may not support live WebSockets."
        )
        print(f"[ERROR] {err_msg}")
        await safe_send_client(client_ws, {"type": "error", "message": err_msg})
    except Exception as ex:  # noqa: BLE001
        err_msg = f"Upstream connection failed: {ex}"
        print(f"[ERROR] {err_msg}")
        await safe_send_client(client_ws, {"type": "error", "message": err_msg})
    finally:
        try:
            await client_ws.close()
        except Exception:
            pass
        print("[INFO] Session ended.")
