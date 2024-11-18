import os
from typing import Any

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler

from snow_classifier.run import run_model


# Define the start command handler
async def start(update: Update, context: Any) -> None:
    run_model()
    await update.message.reply_text(
        "Hello! I'm your simple bot. Send me a message and I'll echo it back!"
    )


def main() -> None:
    # Replace 'YOUR_BOT_TOKEN' with your actual bot token from BotFather
    bot_token = os.environ["BOT_TOKEN"]

    # Create the application instance
    app = ApplicationBuilder().token(bot_token).build()

    # Add handlers to the application
    app.add_handler(CommandHandler("start", start))

    # Start the bot
    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
