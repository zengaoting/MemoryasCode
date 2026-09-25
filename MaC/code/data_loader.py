import re


def load_dataset(path: str) -> list[dict]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unexpected dataset format: {type(data)}")


def get_sessions(sample: dict) -> list[dict]:
    conv = sample.get("conversation", {})
    sessions = []
    for key in conv:
        m = re.match(r"^session_(\d+)$", key)
        if not m:
            continue
        session_num = int(m.group(1))
        datetime_key = f"session_{session_num}_date_time"
        sessions.append(
            {
                "session_id": key,
                "session_number": session_num,
                "datetime": conv.get(datetime_key, ""),
                "messages": conv[key],
            }
        )
    sessions.sort(key=lambda s: s["session_number"])
    return sessions


def format_session_messages(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        dia_id = msg.get("dia_id", "")
        speaker = msg.get("speaker", "Unknown")
        text = msg.get("text", "")
        parts = []
        if dia_id:
            parts.append(f"[{dia_id}]")
        parts.append(f"{speaker}:")
        if text:
            parts.append(text)
        lines.append(" ".join(parts))

        blip = msg.get("blip_caption", "")
        if blip:
            lines.append(f"  Image caption: {blip}")

        query = msg.get("query", "")
        if query:
            lines.append(f"  Image query: {query}")

        img_urls = msg.get("img_url", [])
        for url in img_urls:
            lines.append(f"  Image URL: {url}")

    return "\n".join(lines)
