#!/usr/bin/env python
# pylint: disable=unused-argument
# This program is dedicated to the public domain under the CC0 license.

"""
Telegram bot for generating fake news on a selected topic.
"""
import asyncio
import logging
import string
from concurrent.futures import ThreadPoolExecutor

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, CallbackContext

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

pool = ThreadPoolExecutor(max_workers=4)
loop = asyncio.get_event_loop()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a start message."""
    await update.message.reply_text("hi")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays info on how to use the bot."""
    await update.message.reply_text(
        "Use /start to test this bot.\nUse /generate to generate more fake news headlines."
    )


keyboard = [
    [InlineKeyboardButton("World News", callback_data='WORLD NEWS')],
    [InlineKeyboardButton("Groups Voices", callback_data='GROUPS VOICES')],
    [InlineKeyboardButton("Arts & Culture", callback_data='ARTS & CULTURE')],
    [InlineKeyboardButton("Business & Finances", callback_data='BUSINESS & FINANCES')],
    [InlineKeyboardButton("Science & Tech", callback_data='SCIENCE & TECH')],
    [InlineKeyboardButton("Style & Beauty", callback_data='STYLE & BEAUTY')],
    [InlineKeyboardButton("Панорама", callback_data='panorama')],
]
display_buttons_rows = 7


def init_tok_model(model_path_, model_name_):
    model_ = GPT2LMHeadModel.from_pretrained(
        model_path_)
    tokenizer_ = GPT2Tokenizer.from_pretrained(model_name_)
    tokenizer_.pad_token = tokenizer_.eos_token

    return model_, tokenizer_


model_path = "../2_lora_weights"
model_name = "gpt2"
model, tokenizer = init_tok_model(model_path, model_name)

ru_model_path = "../5_lora_weights"
ru_model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
ru_model, ru_tokenizer = init_tok_model(ru_model_path, ru_model_name)

russian_alphabet = "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"


def generate_headline_sync(model_, tokenizer_, prompt):
    input_ids = tokenizer_.encode(prompt, return_tensors="pt")
    output = model_.generate(input_ids, max_length=30, num_beams=10, no_repeat_ngram_size=2, early_stopping=True,
                             do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    headline = tokenizer_.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    headline = headline.lstrip(" ,.:?!").capitalize()  # capitalize

    headline = "".join([str(char) for char in headline if char in string.printable + russian_alphabet])
    # clean string

    return headline


async def generate_headline(prompt):
    if prompt == 'panorama':
        model_ = ru_model
        tokenizer_ = ru_tokenizer
    else:
        model_ = model
        tokenizer_ = tokenizer
    return await loop.run_in_executor(pool, generate_headline_sync, model_, tokenizer_, prompt)


def run_generation(prompt):
    asyncio.run_coroutine_threadsafe(generate_headline(prompt), loop)


async def generate_buttons(update: Update, context: CallbackContext) -> None:
    """Displays the list of topics for news generation."""
    reply_markup = InlineKeyboardMarkup(keyboard[0:display_buttons_rows])
    await update.message.reply_text("Please choose the topic:", reply_markup=reply_markup)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(text=f"Your headline:\n{await generate_headline(query.data)}")


def main() -> None:
    """Run the bot."""
    token = 'your token here'
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button, block=False))
    application.add_handler(CommandHandler("generate", generate_buttons))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
