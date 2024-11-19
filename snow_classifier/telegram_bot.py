import logging
import os
import sys
from functools import partial

from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from snow_classifier.run import run_model
from snow_classifier.utils import image2buffer

logger = logging.getLogger("snow_classifier")


async def post_init(admin_id: int, application: Application) -> None:
    await application.bot.send_message(
        chat_id=admin_id, text="🟢 The bot has started successfully"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prediction = run_model()
    io_buf = image2buffer(prediction["image"])
    caption = f"Predicted class for {prediction['date']} at {prediction['time']}: {prediction['result']}"

    await update.message.reply_photo(photo=io_buf, caption=caption)


async def restart(
    admin_id: int, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    user_id = update.effective_user.id
    if user_id != admin_id:
        return

    await update.message.reply_text("🔄 Restarting the bot...")
    logger.info(f"Admin ({admin_id}) requested a restart. Restarting the bot...")

    os.execv(sys.executable, ["python", *sys.argv])


async def shutdown(
    admin_id: int, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    user_id = update.effective_user.id
    if user_id != admin_id:
        return

    await update.message.reply_text("🔴 Shutting down the bot...")
    logger.info(f"Admin ({admin_id}) requested a shutdown. Shutting down the bot...")

    context.application.stop_running()


def main() -> None:
    bot_token = os.environ["BOT_TOKEN"]
    admin_id = int(os.environ["ADMIN_ID"])

    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start))

    # Admin handler
    shutdown_func = partial(shutdown, admin_id)
    restart_func = partial(restart, admin_id)
    post_init_func = partial(post_init, admin_id)
    app.add_handler(CommandHandler("shutdown", shutdown_func))
    app.add_handler(CommandHandler("restart", restart_func))
    app.post_init = post_init_func

    # Start the bot
    logger.info("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
