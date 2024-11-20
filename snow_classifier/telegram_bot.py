import logging
import os

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

logger = logging.getLogger("snow_classifier")


class TelegramBot:
    def __init__(self) -> None:
        self.token = os.environ["BOT_TOKEN"]
        self.admin_id = int(os.environ["ADMIN_ID"])
        self.restart_requested = False

    async def post_init(self, application: Application) -> None:
        await application.bot.send_message(
            chat_id=self.admin_id, text="🟢 The bot has started successfully"
        )

    async def warm_shutdown(self) -> None:
        self.app.stop_running()

    async def restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user.id == self.admin_id:
            logger.info(
                f"Admin ({self.admin_id}) requested a restart. Restarting the bot..."
            )
            await update.message.reply_text("🔄 Restarting the bot...")

            self.restart_requested = True
            await self.warm_shutdown()

    async def shutdown(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.effective_user.id == self.admin_id:
            logger.info(
                f"Admin ({self.admin_id}) requested a shutdown. Shutting down the bot..."
            )
            await self.app.bot.send_message(
                chat_id=self.admin_id, text="🔴 Shutting down the bot..."
            )
            await self.warm_shutdown()

    async def admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user.id == self.admin_id:
            keyboard = [
                [
                    InlineKeyboardButton("Restart", callback_data="admin_restart"),
                    InlineKeyboardButton("Shutdown", callback_data="admin_shutdown"),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "Select an option:", reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                "You are not authorized to use this command."
            )

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        if update.effective_user.id != self.admin_id:
            await query.edit_message_text(text="You are not authorized to use this.")
            return

        if query.data == "admin_restart":
            logger.info(
                f"Admin ({self.admin_id}) requested a restart. Restarting the bot..."
            )
            await query.edit_message_text(text="🔄 Restarting the bot...")
            self.restart_requested = True
            await self.warm_shutdown()
        elif query.data == "admin_shutdown":
            logger.info(
                f"Admin ({self.admin_id}) requested a shutdown. Shutting down the bot..."
            )
            await query.edit_message_text(text="🔴 Shutting down the bot...")
            await self.warm_shutdown()

    def run(self) -> None:
        self.restart_requested = True

        while self.restart_requested:
            self.restart_requested = False
            self.app = ApplicationBuilder().token(self.token).build()

            # Admin handlers
            self.app.add_handler(CommandHandler("admin", self.admin))
            self.app.add_handler(CommandHandler("shutdown", self.shutdown))
            self.app.add_handler(CommandHandler("restart", self.restart))
            self.app.add_handler(CallbackQueryHandler(self.button))
            self.app.post_init = self.post_init

            # Start the bot
            logger.info("Bot is running... Press Ctrl+C to stop.")
            self.app.run_polling(close_loop=False)


def run() -> None:
    bot = TelegramBot()
    bot.run()


def run_daemon() -> None:
    import daemon

    with daemon.DaemonContext():
        bot = TelegramBot()
        bot.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    run_daemon()
