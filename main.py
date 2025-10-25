# Minimal Telegram Bot (base level) — PTB 21 + Python 3.13 safe
# Commands: /start, /help, /ping
# Also echoes plain text to prove polling works (DM or group).
from __future__ import annotations
import os, logging, asyncio, signal, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("bot")

# ---------------- Keep-alive HTTP (only if Railway sets $PORT, i.e., service type = Web) -----------
def start_keepalive_if_needed():
    port_env = os.getenv("PORT")
    if not port_env:
        log.info("No $PORT detected — assuming Worker service.")
        return
    try:
        port = int(port_env)
    except Exception:
        log.warning("Invalid PORT=%r; skipping keep-alive server", port_env)
        return

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
        def log_message(self, *a, **k): return

    def run():
        log.info("Keep-alive HTTP listening on 0.0.0.0:%d", port)
        HTTPServer(("0.0.0.0", port), H).serve_forever()

    threading.Thread(target=run, daemon=True).start()

# ---------------- Handlers ----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Bot is alive!\n"
        "Commands:\n"
        "/start — this message\n"
        "/help — help text\n"
        "/ping — quick health check\n\n"
        "Tip: In groups with privacy mode ON, use /ping@<your_bot_username>."
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Help:\n"
        "• DM me /ping to verify responsiveness.\n"
        "• In groups, use /ping@<your_bot_username> if privacy mode is ON."
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

# Echo plain text (proves polling is actually receiving updates)
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text and not update.message.text.startswith("/"):
        await update.message.reply_text("✅ I’m alive and I see your message.")

# Unknown commands => point to /help
async def on_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text and update.message.text.startswith("/"):
        await update.message.reply_text("Unknown command. Try /help")

# ---------------- Build ----------------
def build_app() -> Application:
    token = (
        os.getenv("BOT_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM_TOKEN")
    )
    if not token:
        raise RuntimeError("Set BOT_TOKEN (or TELEGRAM_BOT_TOKEN / TELEGRAM_TOKEN) env var to your Telegram bot token.")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("ping",  cmd_ping))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.COMMAND, on_unknown))
    return app

# ---------------- Manual startup (Python 3.13 friendly) ----------------
async def _main():
    start_keepalive_if_needed()
    app = build_app()

    # Make sure no old webhook blocks polling
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        me = await app.bot.get_me()
        log.info("Authorized as @%s (id=%s)", me.username, me.id)
    except Exception as e:
        log.warning("Startup preflight warning: %s", e)

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    log.info("Polling started.")

    # Idle until SIGINT/SIGTERM (Railway sends SIGTERM on redeploy/stop)
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
    log.info("Starting minimal bot …")
    asyncio.run(_main())
