"""Telegram notification service for door detection alerts."""

import os
import asyncio
from pathlib import Path
from typing import Optional
from telegram import Bot
from telegram.error import TelegramError

# Environment configuration
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


class DoorNotifier:
    """
    Sends Telegram notifications when someone is detected at the door.
    """
    
    def __init__(self, token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.token = token
        self.chat_id = chat_id
        self._bot: Optional[Bot] = None
        
        if not self.token or not self.chat_id:
            print("WARNING: Telegram credentials not set. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
    
    @property
    def bot(self) -> Optional[Bot]:
        if self._bot is None and self.token:
            self._bot = Bot(token=self.token)
        return self._bot
    
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)
    
    async def _send_message_async(self, text: str) -> bool:
        """Send a text message asynchronously."""
        if not self.is_configured():
            print(f"Telegram not configured. Would send: {text}")
            return False
        
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
            return True
        except TelegramError as e:
            print(f"Telegram error: {e}")
            return False
    
    async def _send_photo_async(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo asynchronously."""
        if not self.is_configured():
            print(f"Telegram not configured. Would send photo: {photo_path}")
            return False
        
        try:
            with open(photo_path, "rb") as photo:
                await self.bot.send_photo(chat_id=self.chat_id, photo=photo, caption=caption)
            return True
        except (TelegramError, FileNotFoundError) as e:
            print(f"Telegram error: {e}")
            return False
    
    def send_message(self, text: str) -> bool:
        """Send a text message (blocking)."""
        return asyncio.run(self._send_message_async(text))
    
    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo (blocking)."""
        return asyncio.run(self._send_photo_async(photo_path, caption))
    
    def notify_person(self, name: str, similarity: float, photo_path: Optional[str] = None) -> bool:
        """
        Send notification that a known person was detected.
        
        Args:
            name: Name of the recognized person
            similarity: Recognition confidence (0-1)
            photo_path: Optional path to the detected face image
        """
        message = f"🚪 {name} is at the door! (confidence: {similarity:.0%})"
        
        if photo_path and os.path.exists(photo_path):
            return self.send_photo(photo_path, caption=message)
        else:
            return self.send_message(message)
    
    def notify_unknown(self, similarity: float, photo_path: Optional[str] = None) -> bool:
        """
        Send notification that an unknown person was detected.
        
        Args:
            similarity: Best match confidence (0-1)
            photo_path: Optional path to the detected face image
        """
        message = f"🚪 Unknown visitor at the door (best match: {similarity:.0%})"
        
        if photo_path and os.path.exists(photo_path):
            return self.send_photo(photo_path, caption=message)
        else:
            return self.send_message(message)
