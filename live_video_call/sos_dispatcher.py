"""
Real-Time Emergency SOS Dispatcher for Naythr Assistive AI
==========================================================
Provides:
1. Real-time live network/GPS geolocation lookup with Google Maps link generation.
2. Emergency Email dispatch (SMTP/TLS, HTML + plain text with coordinates and maps pin).
3. Emergency Mobile SMS dispatch (Twilio REST API, Webhooks, or simulated dispatch).
"""

import os
import sys
import json
import logging
import asyncio
import smtplib
import urllib.request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("SosDispatcher")


def get_live_location() -> Dict[str, Any]:
    """Fetch current real-time coordinates and location details.
    
    Uses high-speed network geolocation (accurate to city/neighborhood coordinates).
    Generates a ready-to-click Google Maps pin URL.
    """
    # 1. Primary: ip-api.com
    try:
        req = urllib.request.Request(
            "http://ip-api.com/json",
            headers={"User-Agent": "Naythr-Assistive-SOS/1.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data.get("status") == "success":
                lat = data.get("lat")
                lon = data.get("lon")
                city = data.get("city", "Unknown")
                region = data.get("regionName", "")
                country = data.get("country", "")
                postal = data.get("zip", "")
                maps_url = f"https://www.google.com/maps?q={lat},{lon}"
                address = f"{city}, {region}, {country}".strip(", ")
                return {
                    "lat": lat,
                    "lon": lon,
                    "city": city,
                    "region": region,
                    "country": country,
                    "postal": postal,
                    "maps_url": maps_url,
                    "address": address,
                    "source": "ip-api",
                }
    except Exception as e:
        logger.debug(f"ip-api lookup error: {e}")

    # 2. Fallback: ipinfo.io
    try:
        req = urllib.request.Request(
            "https://ipinfo.io/json",
            headers={"User-Agent": "Naythr-Assistive-SOS/1.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            loc = data.get("loc", "")
            if loc and "," in loc:
                lat, lon = [float(x.strip()) for x in loc.split(",", 1)]
                city = data.get("city", "Unknown")
                region = data.get("region", "")
                country = data.get("country", "")
                postal = data.get("postal", "")
                maps_url = f"https://www.google.com/maps?q={lat},{lon}"
                address = f"{city}, {region}, {country}".strip(", ")
                return {
                    "lat": lat,
                    "lon": lon,
                    "city": city,
                    "region": region,
                    "country": country,
                    "postal": postal,
                    "maps_url": maps_url,
                    "address": address,
                    "source": "ipinfo",
                }
    except Exception as e:
        logger.debug(f"ipinfo lookup error: {e}")

    # 3. Default fallback if offline
    return {
        "lat": 0.0,
        "lon": 0.0,
        "city": "Unknown",
        "region": "Unknown",
        "country": "Unknown",
        "postal": "",
        "maps_url": "https://maps.google.com",
        "address": "Location unavailable (Offline)",
        "source": "fallback",
    }


def send_emergency_email(
    location: Dict[str, Any],
    reason: str = "User indicated emergency / Camera blackout",
    sensor_distances: Optional[Dict[str, Optional[float]]] = None,
) -> bool:
    """Send emergency alert email to configured recipient(s)."""
    to_email = os.getenv("EMERGENCY_EMAIL_TO", "").strip()
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    lat = location.get("lat")
    lon = location.get("lon")
    maps_url = location.get("maps_url", "")
    address = location.get("address", "Unknown Location")
    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    # Sensor summary
    sensor_info = ""
    if sensor_distances:
        sensor_info = " | ".join(
            f"{k}: {v:.0f}mm" if v is not None else f"{k}: N/A"
            for k, v in sensor_distances.items()
        )
    else:
        sensor_info = "Camera Dark Trigger"

    if not to_email or not smtp_user or not smtp_pass:
        print("\n" + "-" * 60)
        print("[SOS EMAIL DISPATCH] (Simulated / Configuration Notice)")
        print(f"  Target Recipient: {to_email or '[Not configured in .env]'}")
        print(f"  Alert Subject:    🚨 EMERGENCY ALERT: Naythr SOS Triggered!")
        print(f"  Live Location:    {address} ({lat}, {lon})")
        print(f"  Google Maps:      {maps_url}")
        print(f"  Trigger Reason:   {reason}")
        print(f"  Sensor Stats:     {sensor_info}")
        print("  -> TIP: Set EMERGENCY_EMAIL_TO, SMTP_USER, and SMTP_PASSWORD in .env to send real emails.")
        print("-" * 60 + "\n")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 EMERGENCY ALERT: Naythr SOS Triggered at {address}"
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["X-Priority"] = "1"  # High priority flag

        text_content = (
            f"EMERGENCY SOS ALERT!\n\n"
            f"The Naythr assistive wearable user has triggered an EMERGENCY SOS.\n\n"
            f"TIME: {timestamp}\n"
            f"REASON: {reason}\n"
            f"LOCATION: {address}\n"
            f"COORDINATES: {lat}, {lon}\n"
            f"GOOGLE MAPS PIN: {maps_url}\n"
            f"TELEMETRY: {sensor_info}\n\n"
            f"Please check on the user immediately!\n"
        )

        html_content = f"""<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; background-color: #f7f9fb; padding: 20px;">
  <div style="max-width: 600px; margin: auto; background: #ffffff; border: 3px solid #d9534f; border-radius: 10px; overflow: hidden;">
    <div style="background-color: #d9534f; color: #ffffff; padding: 18px 24px; font-size: 22px; font-weight: bold; text-align: center;">
      🚨 EMERGENCY SOS ALERT
    </div>
    <div style="padding: 24px; color: #333333; line-height: 1.6;">
      <p style="font-size: 16px;"><strong>The Naythr assistive device has triggered an emergency alert.</strong></p>
      <hr style="border: 0; border-top: 1px solid #eeeeee;">
      <table style="width: 100%; border-collapse: collapse; margin-top: 12px;">
        <tr><td style="padding: 6px 0; color: #777;"><strong>Timestamp:</strong></td><td>{timestamp}</td></tr>
        <tr><td style="padding: 6px 0; color: #777;"><strong>Reason:</strong></td><td style="color: #d9534f; font-weight: bold;">{reason}</td></tr>
        <tr><td style="padding: 6px 0; color: #777;"><strong>Live Location:</strong></td><td>{address}</td></tr>
        <tr><td style="padding: 6px 0; color: #777;"><strong>Coordinates:</strong></td><td>{lat}, {lon}</td></tr>
        <tr><td style="padding: 6px 0; color: #777;"><strong>Wearable Sensors:</strong></td><td>{sensor_info}</td></tr>
      </table>
      <div style="text-align: center; margin: 24px 0;">
        <a href="{maps_url}" target="_blank" style="background-color: #0275d8; color: #ffffff; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 16px; display: inline-block;">
          📍 Open Live Google Maps Location
        </a>
      </div>
      <p style="font-size: 13px; color: #888888; text-align: center;">Please contact the user or verify their safety immediately.</p>
    </div>
  </div>
</body>
</html>
"""
        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        recipients = [e.strip() for e in to_email.split(",") if e.strip()]
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipients, msg.as_string())

        print(f"\n[SOS EMAIL] Emergency email alert successfully sent to: {', '.join(recipients)}!\n")
        return True
    except Exception as e:
        print(f"\n[SOS EMAIL ERROR] Failed to send emergency email: {e}\n")
        return False


def send_emergency_sms(
    location: Dict[str, Any],
    reason: str = "User indicated emergency / Camera blackout",
) -> bool:
    """Send emergency SMS alert to user's mobile phone number."""
    phone_number = os.getenv("EMERGENCY_PHONE_NUMBER", "").strip()
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    twilio_from = os.getenv("TWILIO_FROM_NUMBER", "").strip()
    webhook_url = os.getenv("EMERGENCY_SMS_WEBHOOK_URL", "").strip()

    lat = location.get("lat")
    lon = location.get("lon")
    maps_url = location.get("maps_url", "")
    address = location.get("address", "Unknown Location")

    sms_body = (
        f"🚨 EMERGENCY ALERT: Naythr user triggered SOS!\n"
        f"Reason: {reason}\n"
        f"Location: {address} ({lat}, {lon})\n"
        f"Map: {maps_url}\n"
        f"Please check on them immediately!"
    )

    # 1. Twilio REST API (direct HTTP POST without needing twilio package)
    if phone_number and twilio_sid and twilio_token and twilio_from and requests:
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
            res = requests.post(
                url,
                data={
                    "From": twilio_from,
                    "To": phone_number,
                    "Body": sms_body,
                },
                auth=(twilio_sid, twilio_token),
                timeout=8,
            )
            if res.status_code in (200, 201):
                res_data = res.json()
                print(f"\n[SOS SMS] Twilio SMS sent to {phone_number} (SID: {res_data.get('sid')})!\n")
                return True
            else:
                print(f"\n[SOS SMS ERROR] Twilio returned status {res.status_code}: {res.text}\n")
        except Exception as e:
            print(f"\n[SOS SMS ERROR] Failed to send Twilio SMS: {e}\n")

    # 2. Webhook SMS Gateway (e.g. CallMeBot, Telegram bot, or Fast2SMS)
    if webhook_url and requests:
        try:
            payload = {
                "event": "EMERGENCY_SOS",
                "phone": phone_number,
                "reason": reason,
                "location": location,
                "message": sms_body,
            }
            res = requests.post(webhook_url, json=payload, timeout=8)
            print(f"[SOS WEBHOOK] Dispatched emergency payload to {webhook_url} (Status: {res.status_code})")
            return True
        except Exception as e:
            print(f"[SOS WEBHOOK ERROR] {e}")

    # 3. Always log simulated SMS dispatch so the user sees the output and live link
    print("\n" + "-" * 60)
    print("[SOS MOBILE SMS DISPATCH] (Simulated / Ready for Gateway)")
    print(f"  Target Mobile:  {phone_number or '[Not configured in .env]'}")
    print(f"  Text Message:   {sms_body.replace(chr(10), ' | ')}")
    print(f"  Google Maps:    {maps_url}")
    print("  -> TIP: Set EMERGENCY_PHONE_NUMBER and TWILIO_ACCOUNT_SID/AUTH_TOKEN in .env for automated SMS.")
    print("-" * 60 + "\n")
    return False


async def dispatch_all_emergency_alerts(
    location: Optional[Dict[str, Any]] = None,
    reason: str = "User indicated emergency / Camera blackout",
    sensor_distances: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    """Execute complete multi-channel emergency alert dispatch asynchronously."""
    # 1. Acquire live coordinates if not already passed
    if location is None:
        loop = asyncio.get_running_loop()
        location = await loop.run_in_executor(None, get_live_location)

    # 2. Dispatch email and SMS in parallel without blocking main thread
    loop = asyncio.get_running_loop()
    email_future = loop.run_in_executor(
        None, send_emergency_email, location, reason, sensor_distances
    )
    sms_future = loop.run_in_executor(
        None, send_emergency_sms, location, reason
    )

    email_sent, sms_sent = await asyncio.gather(email_future, sms_future, return_exceptions=True)

    return {
        "location": location,
        "email_sent": bool(email_sent is True),
        "sms_sent": bool(sms_sent is True),
    }


if __name__ == "__main__":
    print("Testing Live Geolocation & Alert Dispatch...")
    loc = get_live_location()
    print("Acquired Location:", json.dumps(loc, indent=2))
    asyncio.run(dispatch_all_emergency_alerts(loc, reason="Unit Test Verification"))
