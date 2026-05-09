"""
POST /api/v1/contact
--------------------
Receives a contact form payload, validates it via Pydantic, and constructs 
a structured plain-text email body to bypass strict spam filters.

Dispatches the message via Gmail SMTP to the team leader and CCs team members.
Logs all requests and dispatch statuses using the internal Logger.
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter, HTTPException, Request
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType

from backend.app.limiter import limiter

from backend.utils.logger import Logger

load_dotenv()

# Initialize the custom logger
logger = Logger()

router = APIRouter(prefix="/api/v1", tags=["Contact"])

# Configuration (Gmail SMTP)
_REQUIRED_ENV = ("MAIL_USERNAME", "MAIL_PASSWORD", "MAIL_FROM", "TEAM_LEADER")
_missing_env = [var for var in _REQUIRED_ENV if not os.getenv(var)]

if _missing_env:
    logger.warning(
        f"Email feature DISABLED — missing env vars: {', '.join(_missing_env)}. "
        "Contact form will accept submissions but will not dispatch emails."
    )
    conf = None
else:
    conf = ConnectionConfig(
        MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
        MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
        MAIL_FROM=os.getenv("MAIL_FROM"),
        MAIL_PORT=465,
        MAIL_SERVER="smtp.gmail.com",
        MAIL_STARTTLS=False,
        MAIL_SSL_TLS=True,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True,
    )

class ContactForm(BaseModel):
    # Field constraints handle length and reject empty strings
    name: str = Field(..., min_length=2, max_length=100)
    subject: str = Field(..., min_length=3, max_length=150)
    message: str = Field(..., min_length=10, max_length=3000)
    
    @field_validator('name', 'subject', mode='before')
    @classmethod
    def strip_newlines(cls, v: str) -> str:
        return v.replace('\n', ' ').replace('\r', ' ')
    
    model_config = {"str_strip_whitespace": True}

def sanitize_for_log(value: str, max_len: int = 100) -> str:
    return value.replace('\n', '\\n').replace('\r', '\\r')[:max_len]

@router.post("/contact")
@limiter.limit("10/minute")
async def handle_contact_form(request: Request, form: ContactForm):
    # Log the start of the request
    logger.info(f"Contact form from: {sanitize_for_log(form.name)} | Subject: {sanitize_for_log(form.subject)}")
    
    if conf is None:
        logger.warning(
            f"Email skipped (env not configured) for: {sanitize_for_log(form.name)}"
        )
        # Still return 200 — the form worked, email just isn't available
        return {
            "status": "received",
            "detail": "Message received. Email dispatch is currently unavailable.",
        }

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

    email_subject = form.subject

    message = MessageSchema(
        subject=email_subject,
        recipients=[os.getenv("TEAM_LEADER")],
        cc=os.getenv("TEAM_MEMBERS").split(",") if os.getenv("TEAM_MEMBERS") else [],
        body=structured_body,
        subtype=MessageType.plain  # Keep as plain for maximum delivery
    )

    fm = FastMail(conf)

    try:
        # Attempt to send the email
        await fm.send_message(message)

        # Log successful dispatch
        logger.info(f"Email successfully sent for: {sanitize_for_log(form.name)}")
        return {"status": "success"}

    except Exception as e:
        # Log the error with a stack trace via your custom Logger
        logger.error(f"Failed to send contact email for {sanitize_for_log(form.name)}: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process request at this time. Please try again later."
        )