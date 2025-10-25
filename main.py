
# Minimal Telegram Bot (base level) — PTB 21 + Python 3.13 safe
# Features: /start, /help, /ping, unknown-command handler, clean manual startup
from __future__ import annotations
import os, logging, asyncio, signal
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("bot")

# ---- Handlers ----
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Bot is alive!\n"
        "Commands:\n"
        "/start — this message\n"
        "/help — help text\n"
        "/ping — quick health check"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Help:\n"
        "• Add me to a group or DM me here.\n"
        "• Try /ping to verify responsiveness."
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        await update.message.reply_text("Unknown command. Try /help")

# ---- Build ----
def build_app() -> Application:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Set BOT_TOKEN env var to your Telegram Bot token.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("ping",  cmd_ping))
    app.add_handler(MessageHandler(filters.COMMAND, unknown))
    return app

# ---- Manual startup (Python 3.13 friendly) ----
async def _main():
    app = build_app()
    try:
        # Ensure webhook is removed so polling works
        await app.bot.delete_webhook(drop_pending_updates=True)
        me = await app.bot.get_me()
        log.info("Authorized as @%s (id=%s)", me.username, me.id)
    except Exception as e:
        log.warning("Startup preflight warning: %s", e)

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    # Idle until SIGINT/SIGTERM
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    await stop.wait()

    try:
        await app.updater.stop()
    except Exception:
        pass
    await app.stop()
    await app.shutdown()

if __name__ == "__main__":
    print("Starting minimal bot …")
    asyncio.run(_main())
