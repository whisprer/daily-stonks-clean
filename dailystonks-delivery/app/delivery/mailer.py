from __future__ import annotations

import smtplib
from email.message import EmailMessage
from email.utils import make_msgid

from ..config import Settings

def send_email(
    settings: Settings,
    to_email: str,
    subject: str,
    html_body: str,
    list_unsubscribe_url: str,
) -> str:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.mail_from
    msg["To"] = to_email
    msg["Message-ID"] = make_msgid(domain=None)
    msg["MIME-Version"] = "1.0"
    msg["List-Unsubscribe"] = f"<{list_unsubscribe_url}>"
    msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"

    msg.set_content("Your email client does not support HTML.")
    msg.add_alternative(html_body, subtype="html")

    if settings.smtp_use_tls:
        server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)
        server.starttls()
    else:
        server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)

    try:
        if settings.smtp_username and settings.smtp_password:
            server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(msg)
    finally:
        try:
            server.quit()
        except Exception:
            pass

    return msg["Message-ID"]
