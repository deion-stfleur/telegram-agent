import os
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import caldav
from dotenv import load_dotenv
from imap_tools import AND, MailBox

load_dotenv()


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in .env")
    return value


def _mailbox() -> MailBox:
    address = _required("GMAIL_ADDRESS")
    password = _required("GMAIL_APP_PASSWORD").replace(" ", "")
    host = os.environ.get("IMAP_HOST", "imap.gmail.com")
    return MailBox(host).login(address, password)


def _as_dict(msg, body_limit: int) -> dict:
    return {
        "uid": str(msg.uid),
        "from": msg.from_ or "",
        "subject": msg.subject or "(no subject)",
        "date": msg.date.isoformat() if msg.date else "",
        "body": (msg.text or msg.html or "")[:body_limit],
    }


def fetch_unread_emails(limit: int = 5) -> list[dict]:
    """Return unread inbox messages as dicts with uid, from, subject, and body."""
    emails = []
    with _mailbox() as mailbox:
        for msg in mailbox.fetch(AND(seen=False), limit=limit, reverse=True, mark_seen=False):
            emails.append(_as_dict(msg, 8000))
    return emails


def search_emails(query: str, limit: int = 8) -> str:
    """Search the inbox by sender, subject, or body. Does not mark mail as read."""
    query = (query or "").strip()
    try:
        with _mailbox() as mailbox:
            criteria = AND(text=query) if query else "ALL"
            found = list(mailbox.fetch(criteria, limit=limit, reverse=True, mark_seen=False, charset="UTF-8"))
            if query and len(found) < limit:
                seen = {msg.uid for msg in found}
                needle = query.lower()
                for msg in mailbox.fetch(limit=40, reverse=True, mark_seen=False):
                    if msg.uid in seen:
                        continue
                    blob = f"{msg.from_}\n{msg.subject}\n{msg.text}".lower()
                    if needle in blob:
                        found.append(msg)
                    if len(found) >= limit:
                        break
    except Exception as exc:
        return f"Could not search email: {exc}"

    if not found:
        return f'No inbox messages matched "{query}".' if query else "Inbox is empty."

    lines = []
    for msg in found[:limit]:
        item = _as_dict(msg, 700)
        lines.append(
            f"From: {item['from']}\nDate: {item['date']}\nSubject: {item['subject']}\n{item['body']}"
        )
    return "\n\n---\n\n".join(lines)


def _calendar():
    """Google Calendar still accepts the Gmail app password on this CalDAV collection."""
    address = _required("GMAIL_ADDRESS")
    password = _required("GMAIL_APP_PASSWORD").replace(" ", "")
    url = os.environ.get(
        "CALDAV_URL",
        f"https://www.google.com/calendar/dav/{address}/events",
    )
    client = caldav.DAVClient(url=url, username=address, password=password)
    return client.calendar(url=url)


def add_google_calendar_event(
    title: str, start_iso: str, end_iso: str, description: str = ""
) -> str:
    """Create one event on the Google Calendar CalDAV collection. Returns a status string."""
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso) if end_iso else start

    try:
        _calendar().add_event(
            summary=title,
            dtstart=start,
            dtend=end,
            description=description or "",
        )
    except Exception as exc:
        return f"Could not add calendar event: {exc}"
    return f'Added "{title}" ({start_iso} to {end_iso}).'


def list_events_for_date(day) -> str:
    """Return events whose start falls on day in America/New_York."""
    tz = ZoneInfo("America/New_York")
    start = datetime.combine(day, time.min, tzinfo=tz)
    try:
        events = _calendar().search(start=start, end=start + timedelta(days=1), event=True)
    except Exception as exc:
        return f"Could not read calendar: {exc}"

    lines = []
    for event in events:
        component = event.icalendar_component
        title = str(component.get("SUMMARY", "(no title)"))
        dtstart = component.get("DTSTART")
        when = dtstart.dt.isoformat() if dtstart is not None else "?"
        lines.append(f"{when}: {title}")
    return "\n".join(lines) if lines else "No events today."


def list_calendar_events(query: str = "", days_ahead: int = 30) -> str:
    """List calendar events from a week ago through days_ahead. Optional query matches the title."""
    now = datetime.now()
    try:
        events = _calendar().search(
            start=now - timedelta(days=7),
            end=now + timedelta(days=max(1, days_ahead)),
            event=True,
        )
    except Exception as exc:
        return f"Could not read calendar: {exc}"

    needle = (query or "").strip().lower()
    lines = []
    for event in events:
        component = event.icalendar_component
        title = str(component.get("SUMMARY", "(no title)"))
        if needle and needle not in title.lower():
            continue
        start = component.get("DTSTART")
        when = start.dt.isoformat() if start is not None else "?"
        lines.append(f"{when}: {title}")
    if not lines:
        return f'No calendar events matched "{query}".' if needle else "No events in that window."
    return "\n".join(lines[:20])
