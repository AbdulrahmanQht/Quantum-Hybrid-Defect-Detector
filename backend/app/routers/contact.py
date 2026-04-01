import os
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from backend.utils.logger import Logger

load_dotenv()

# Initialize the custom logger
logger = Logger()
router = APIRouter()

# ─── Configuration (Gmail SMTP) ───────────────────────────────────────────
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=465,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=False,
    MAIL_SSL_TLS=True,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)


class ContactForm(BaseModel):
    name: str
    subject: str
    message: str


@router.post("/api/contact")
async def handle_contact_form(form: ContactForm):
    # Log the start of the request
    logger.info(f"Contact form request received from: {form.name} | Subject: {form.subject}")

    # Structured Plain Text Body (Safe for IAU Outlook Filters)
    structured_body = (
        f"--- NEW CONTACT INQUIRY ---\n\n"
        f"TITLE: {form.subject}\n"
        f"FROM: {form.name}\n"
        f"---------------------------\n\n"
        f"MESSAGE:\n"
        f"{form.message}\n\n"
        f"---------------------------\n"
        f"Sent via Graduation Project Automated Bot."
    )

    email_subject = form.subject.strip() if form.subject.strip() else "New Project Inquiry"

    message = MessageSchema(
        subject=email_subject,
        recipients=[os.getenv("TEAM_LEADER")],
        cc=os.getenv("TEAM_MEMBERS").split(","),
        body=structured_body,
        subtype=MessageType.plain  # Keep as plain for maximum delivery
    )

    fm = FastMail(conf)

    try:
        # Attempt to send the email
        await fm.send_message(message)

        # Log successful dispatch
        logger.info(f"Email successfully sent for: {form.name}")
        return {"status": "success"}

    except Exception as e:
        # Log the error with a stack trace via your custom Logger
        logger.error(f"Failed to send contact email for {form.name}: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error during email dispatch"
        )