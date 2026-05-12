import anthropic
from anthropic.types import ToolUseBlock
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from pydub import AudioSegment
import io
import json
from supabase import create_client
import boto3
from botocore.config import Config

load_dotenv()

client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

r2 = boto3.client(
    "s3",
    endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    config=Config(signature_version="s3v4"),
    region_name="auto"
)

male_voices   = ["fable", "onyx", "echo"]
female_voices = ["nova", "shimmer", "alloy"]
male_index    = 0
female_index  = 0
voice_map = {}

#────────────────────────────────Helpers────────────────────────────────────────────────

def parse_script(script: str):
    lines = script.split("\n")
    parsed = []
    current_speaker = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # skip formatting lines
        if line.startswith("#") or line.startswith("---") or line.lower().startswith("you will hear"):
            continue

        # detect speaker line: **[M] Name:** or **[F] Name:**
        if line.startswith("**[M]") or line.startswith("**[F]"):
            gender = line[3]  # 'M' or 'F'
            colon_pos = line.find(":**")
            if colon_pos != -1:
                name = line[5:colon_pos].strip()  # text between **[M] and :**
                current_speaker = f"[{gender}] {name}"
                dialogue = line[colon_pos + 3:].strip()
                dialogue = dialogue.replace("...", " ").replace("*", "").strip()
                if dialogue:
                    parsed.append((current_speaker, dialogue))
                continue

        # continuation line
        if current_speaker:
            dialogue = line.replace("...", " ").replace("*", "").strip()
            if dialogue:
                parsed.append((current_speaker, dialogue))

    return parsed

def get_voice(speaker: str, all_speakers: list):
    global male_index, female_index

    if speaker in voice_map:
        return voice_map[speaker]

    if len(all_speakers) == 1:
        voice = "fable"
        voice_map[speaker] = voice
        return voice

    if speaker.startswith("[M]"):
        voice = male_voices[male_index % len(male_voices)]
        male_index += 1
    elif speaker.startswith("[F]"):
        voice = female_voices[female_index % len(female_voices)]
        female_index += 1
    else:
        voice = "alloy"

    voice_map[speaker] = voice
    return voice

#────────────────────────────────Agent────────────────────────────────────────────────

def script_agent(section : str, topic : str):

    messages = [{"role": "user", "content" : f"Section: {section}\n Topic : {topic}"}]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=f"""You are an expert IELTS script writer. Write an IELTS Listening script for {section} about {topic}.

        STRICT FORMATTING RULES — follow exactly:
        - Every single line of speech MUST start with **[M] Name:** or **[F] Name:**
        - Example: **[M] David:** Good morning, welcome to the museum.
        - Example: **[F] Sarah:** Could you tell us more about the exhibits?
        - NO bare [M] or [F] tags without a name
        - NO markdown headers, NO --- dividers, NO *[pause]* tags
        - Use ... for natural pauses within dialogue
        - British English only
        - Under 3500 characters
        - CEFR B2-C1 vocabulary""",
        messages=messages
    )

    with open("script.txt", "w", encoding="utf-8") as f:
        f.write(response.content[0].text)

    return response.content[0].text

def question_agent(script: str) -> list[dict]:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system="""
        Generate 10 IELTS listening questions for this script. 
        Return ONLY a JSON array, no markdown, no explanation.

        Constraints:
        1. Question Types: Use only "mcq", "fill", or "matching". 
        2. Validity: For "fill", ensure the answer is verbatim from the script. 
        3. Logic: Ensure questions follow the chronological order of the script.
        4. JSON Schema:
            - order_index: (1-10)   
            - question_type: "mcq", "fill", "matching"
            - question_text: The prompt
            - options: 
                * For "mcq": array of 4 answer strings
                * For "matching": array of items to match (left side)
                * For "fill": null
            - matching_pool:
                * For "matching": array of options to match against (right side) e.g. ["Student A", "Student B", "Supervisor"]
                * For others: null
            - answer_key: 
                * For "mcq": numeric index of correct option (0, 1, 2, 3)
                * For "fill": exact word or phrase from the script
                * For "matching": JSON object mapping index to correct match e.g. {"0": "Student A", "1": "Supervisor"}
            - wrong_answer_tip: A brief explanation of why the other options are wrong
        """,
        messages=[{"role": "user", "content": script}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    questions = json.loads(raw)
    for q in questions:
        if q["question_type"] == "tfng":
            q["options"] = ["True", "False", "Not Given"]
            answer_map = {"true": 0, "false": 1, "not given": 2}
            q["answer_key"] = answer_map.get(str(q["answer_key"]).lower(), 0)

        elif q["question_type"] == "mcq":
            if isinstance(q["answer_key"], str):
                if q["answer_key"].isdigit():
                    q["answer_key"] = int(q["answer_key"])
                elif q.get("options"):
                    try:
                        q["answer_key"] = q["options"].index(q["answer_key"])
                    except ValueError:
                        q["answer_key"] = 0
            q["answer_key"] = int(q["answer_key"])
            q["matching_pool"] = None

        elif q["question_type"] == "matching":
            # ensure answer_key is a dict like {"0": "Student A", "1": "Supervisor"}
            if isinstance(q["answer_key"], list):
                q["answer_key"] = {str(i): v for i, v in enumerate(q["answer_key"])}
            elif isinstance(q["answer_key"], str):
                try:
                    q["answer_key"] = json.loads(q["answer_key"])
                except:
                    q["answer_key"] = {}
            # ensure matching_pool exists
            if not q.get("matching_pool"):
                q["matching_pool"] = q.get("options", [])

        elif q["question_type"] == "fill":
            q["options"] = None
            q["matching_pool"] = None

    return questions

def tts_agent(script: str):
    content = parse_script(script)
    audio_segments = []

    all_speakers = list(set(speaker for speaker, _ in content))  # unique speakers
    print(f"Speakers detected: {all_speakers}")

    for speaker, dialogue in content:
        if not dialogue.strip():
            continue

        voice = get_voice(speaker, all_speakers)  # ← pass all_speakers
        clean_name = speaker.replace("[M]", "").replace("[F]", "").strip()
        print(f"{clean_name} ({voice}): {dialogue[:30]}...")

        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=dialogue
        )

        audio_bytes = io.BytesIO(response.content)
        segment = AudioSegment.from_file(audio_bytes, format="mp3")
        pause = AudioSegment.silent(duration=400)
        audio_segments.append(segment + pause)

    combined = sum(audio_segments, AudioSegment.empty())
    output_path = "modified.mp3"
    combined.export(output_path, format="mp3")

    print(f"✅ Audio saved to {output_path} | Voices used: {voice_map}")
    return output_path

#────────────────────────────────Supabase────────────────────────────────────

def get_or_create_test(topic: str) -> str:
    title = test_name
    result = supabase.table("listening_tests").select("id").eq("title", title).execute()
    if result.data:  # test already exists
        return result.data[0]["id"]
    result = supabase.table("listening_tests").insert({
        "title":     test_name,
        "is_active": True,
        "is_demo":   False
    }).execute()
    return result.data[0]["id"]

def save_to_database(test_name : str, section: str, topic: str, script : str, questions : list[dict]):

    test_id = get_or_create_test(test_name)
    section_number = int(re.search(r'\d+', section).group()) if re.search(r'\d+', section) else 1
    with open("modified.mp3", "rb") as f:
        audio_bytes = f.read()
    file_name = f"section{section_number}_{topic}.mp3".replace(" ", "_")
    r2.put_object(
        Bucket=os.getenv("R2_BUCKET_NAME"),
        Key=file_name,
        Body=audio_bytes,
        ContentType="audio/mpeg"
    )
    audio_url = f"{os.getenv('R2_PUBLIC_URL')}/{file_name}"

    # insert listening_sections
    section_row = supabase.table("listening_sections").insert({
        "test_id":  test_id,
        "section_number": section_number,
        "title":          topic,
        "context":        topic,
        "audio_url":      audio_url,
    }).execute()
    section_id = section_row.data[0]["id"]

    # insert listening_questions
    for q in questions:
        supabase.table("listening_questions").insert({
            "section_id":       section_id,
            "order_index":      q["order_index"],
            "question_type":    q["question_type"],
            "question_text":    q["question_text"],
            "options":          q["options"],
            "answer_key":       q["answer_key"],
            "wrong_answer_tip": q["wrong_answer_tip"],
        }).execute()

    print(f"✅ Audio: {audio_url}")
    print(f"✅ Saved — section_id: {section_id}")
    return section_id

#────────────────────────────────Orchestrator────────────────────────────────

orchestrator_tools = [
    {
        "name": "script_agent",
        "description" : "Create a conversation script.",
        "input_schema": {
            "type":"object",
            "properties": {
                "section": {"type":"string", "description":"The section of the question."},
                "topic": {"type":"string", "description":"The topic of the question."}
            },
            "required": ["section","topic"]
        }
    },
    {
        "name": "question_agent",
        "description": "Create questions based on the script.",
        "input_schema": {
            "type":"object",
            "properties": {
                "script": {"type":"string", "description":"The script for the question."}
            },
            "required":["script"]
        }
    },
    {
        "name": "tts_agent",
        "description": "Convert the IELTS script to speech. Always pass the full script generated by script_agent as the input.",
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "The full script text generated by script_agent to convert to audio"}
            },
            "required": ["script"]
        }
    }
]

def orchestrator(section: str, topic: str):
    messages = [{"role": "user", "content": f"Section: {section}\nTopic: {topic}"}]
    system = "You are an orchestrator. Follow this exact order: 1) Call script_agent to generate a script, 2) Call question_agent passing the script, 3) Call tts_agent passing the same script from step 1. Always pass the script text between agents."

    results = {"script": None, "questions": None, "audio": None}  # ← Bug 1 fix

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        tools=orchestrator_tools,
        system=system,
        messages=messages
    )

    while response.stop_reason != "end_turn":
        tool_results = []

        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue

            if block.name == "script_agent":
                results["script"] = script_agent(**block.input)
                content = str(results["script"])
            elif block.name == "question_agent":
                results["questions"] = question_agent(**block.input)
                content = json.dumps(results["questions"])
            elif block.name == "tts_agent":
                results["audio"] = tts_agent(**block.input)
                content = "Audio generated successfully."

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": [{"type": "text", "text": content}]
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            tools=orchestrator_tools,
            system=system,
            messages=messages
        )

    return results  # ← Bug 2 fix: return results dict, not response text

section = "Ielts listening test Section 1: conversation between two people"
topic = "reserving a hotel"
test_name = "Academic Listening Test 3"

results = orchestrator(section, topic)

if all(results.values()):
    save_to_database(test_name,section, topic, results["script"], results["questions"])
    print("✅ Done!")
else:
    print("⚠️ Something didn't complete:", {k: v is not None for k, v in results.items()})