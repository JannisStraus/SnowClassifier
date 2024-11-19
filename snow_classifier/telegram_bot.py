import logging
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from snow_classifier.run import run_model
from snow_classifier.utils import cv2_to_buffer

logger = logging.getLogger("snow_classifier")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prediction = run_model()
    io_buf = cv2_to_buffer(prediction["image"])
    caption = f"Predicted class for {prediction['date']} at {prediction['time']}: {prediction['result']}"

    await update.message.reply_photo(photo=io_buf, caption=caption)


def main() -> None:
    # Replace 'YOUR_BOT_TOKEN' with your actual bot token from BotFather
    bot_token = os.environ["BOT_TOKEN"]

    # Create the application instance
    app = ApplicationBuilder().token(bot_token).build()

    # Add handlers to the application
    app.add_handler(CommandHandler("start", start))

    # Start the bot
    logger.info("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
