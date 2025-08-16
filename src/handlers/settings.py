from typing import Final, Literal, Optional
import html
import logging
from enum import Enum
from asyncio import sleep
from string import Template
from functools import partial

from aiogram import Router, F
from aiogram.enums import ParseMode, ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.methods import GetChatAdministrators
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.utils import BotSettings, AutoValueEnum

PROBABILITY_STEP: Final[int] = 5
MIN_PROBABILITY: Final[int] = 0
MAX_PROBABILITY: Final[int] = 100

logger = logging.getLogger(__name__)


class StringEnum(AutoValueEnum):
    """Secure string constants with safe composition"""
    SETTINGS_TITLE = "⚙️ Настройки бота"
    CHANCE_EXPLANATION = "Чем выше процент, тем больше шанс, что бот будет самостоятельно вступать в диалог"
    BOT_STATUS_EXPLANATION = "<b>Состояние бота:</b> когда бот работает, то вверху чата будет отображаться \"печатает...\""
    ERROR_DISPLAY_EXPLANATION = "<b>Показывать ошибки бота:</b> отправка сообщений с внутренними ошибками бота в чат"

    SHOW_BOT_STATUS = "Показывать состояние бота"
    SHOW_BOT_ERRORS = "Показывать ошибки бота"
    DONE_BUTTON = "Готово"

    ADMIN_ONLY = "⚠️ Эта команда доступна только администраторам группы."
    ADMIN_ONLY_CALLBACK = "⚠️ Это могут делать только администраторы группы."
    SETTINGS_ERROR = "⚠️ Не удалось отобразить настройки. Попробуйте позже."
    UNSUPPORTED_OPERATION = "⚠️ Неподдерживаемая операция"
    SETTINGS_SAVED = "Настройки сохранены!"
    SETTINGS_SAVED_WITH_DELETE_ERROR = "Настройки сохранены! (Сообщение не удалено)"
    MAX_LIMIT_REACHED = "Достигнуто предельное значение"
    PRIVATE_CHAT_UNAVAILABLE = "Эта опция недоступна в приватном чате"
    UNKNOWN_ERROR = "⚠️ Произошла ошибка"

    BOT_STATUS_UPDATED = "Состояние бота теперь $status"
    ERROR_DISPLAY_UPDATED = "Показ ошибок теперь $status"
    CURRENT_PROBABILITY = "Текущая вероятность: $prob%"
    PROBABILITY_SET = "Установлено: $prob%"

    ENABLED = "включен"
    DISABLED = "выключен"
    VISIBLE = "отображается"
    HIDDEN = "скрыто"

    @classmethod
    def compose(
            cls,
            template: Literal[
                'BOT_STATUS_UPDATED',
                'ERROR_DISPLAY_UPDATED',
                'CURRENT_PROBABILITY',
                'PROBABILITY_SET'
            ],
            **kwargs: str
    ) -> str:
        """
        Type-safe string composition using pre-defined templates.
        All values are automatically HTML-escaped.
        """
        template_str = str(cls[template])
        safe_kwargs = {k: html.escape(str(v)) for k, v in kwargs.items()}
        return Template(template_str).substitute(**safe_kwargs)


class SettingsAction(str, Enum):
    """Available actions in settings callback"""
    DECREASE = "less"
    INCREASE = "more"
    CONFIRM = "ok"
    TOGGLE_CHAT_ACTIONS = "toggle_chat"
    TOGGLE_ERRORS = "toggle_errors"
    NOOP = "_"


class SettingsKeyboardBuilder:
    """Builder for settings keyboard with current values"""

    @staticmethod
    def validate_probability(prob: int) -> int:
        """Ensure probability stays within bounds"""
        return max(MIN_PROBABILITY, min(prob, MAX_PROBABILITY))

    @classmethod
    def create(cls, message: Message, bot_storage: BotSettings, chat_id: int) -> InlineKeyboardBuilder:
        """Create interactive settings keyboard"""
        builder = InlineKeyboardBuilder()
        current_prob = bot_storage.get_intervention_probability(chat_id)

        if message.chat.type != ChatType.PRIVATE:
            decrease_action = (
                SettingsAction.NOOP.value
                if current_prob <= MIN_PROBABILITY
                else SettingsAction.DECREASE.value
            )
            increase_action = (
                SettingsAction.NOOP.value
                if current_prob >= MAX_PROBABILITY
                else SettingsAction.INCREASE.value
            )

            builder.row(
                InlineKeyboardButton(
                    text="<<",
                    callback_data=f"settings:{decrease_action}"
                ),
                InlineKeyboardButton(
                    text=f"{current_prob}%",
                    callback_data="settings:_"
                ),
                InlineKeyboardButton(
                    text=">>",
                    callback_data=f"settings:{increase_action}"
                ),
            )

        builder.row(InlineKeyboardButton(
            text=f"{'✅' if bot_storage.get_chat_actions(chat_id) else '❌'} {StringEnum.SHOW_BOT_STATUS}",
            callback_data=f"settings:{SettingsAction.TOGGLE_CHAT_ACTIONS.value}"
        ))

        builder.row(InlineKeyboardButton(
            text=f"{'✅' if bot_storage.get_display_errors(chat_id) else '❌'} {StringEnum.SHOW_BOT_ERRORS}",
            callback_data=f"settings:{SettingsAction.TOGGLE_ERRORS.value}"
        ))

        builder.row(InlineKeyboardButton(
            text=StringEnum.DONE_BUTTON,
            callback_data=f"settings:{SettingsAction.CONFIRM.value}"
        ))

        return builder


router = Router()


async def _is_user_admin(bot, chat_id: int, user_id: int) -> bool:
    """Check if user is admin in the chat"""
    if chat_id == user_id:
        return True

    try:
        admins = await bot(GetChatAdministrators(chat_id=chat_id))
        return user_id in {admin.user.id for admin in admins}
    except (TelegramBadRequest, TelegramForbiddenError) as e:
        logger.warning(f"Failed to check admin status for user {user_id} in chat {chat_id}: {e}")
        return False


async def _send_temporary_message(message: Message, text: str, delay: int = 5) -> None:
    """Send a temporary message that auto-deletes after delay"""
    try:
        msg = await message.answer(text, parse_mode=None)
        await sleep(delay)
        await msg.delete()
    except TelegramBadRequest as e:
        logger.warning(f"Failed to send/delete temporary message: {e}")


async def _handle_settings_show(
        message: Message,
        bot_storage: BotSettings,
        chat_id: Optional[int] = None
) -> None:
    """Show settings interface with proper access checks"""
    chat_id = chat_id or message.chat.id

    if message.chat.type != ChatType.PRIVATE:
        if not await _is_user_admin(message.bot, chat_id, message.from_user.id):
            await _send_temporary_message(message, StringEnum.ADMIN_ONLY)
            return

    keyboard = SettingsKeyboardBuilder.create(message, bot_storage, chat_id)
    chance_text = f"<i>{StringEnum.CHANCE_EXPLANATION}</i>\n" if message.chat.type != ChatType.PRIVATE else ""

    try:
        await message.answer(
            f"<b>{StringEnum.SETTINGS_TITLE}</b>\n\n"
            f"{chance_text}"
            f"{StringEnum.BOT_STATUS_EXPLANATION}\n"
            f"{StringEnum.ERROR_DISPLAY_EXPLANATION}",
            reply_markup=keyboard.as_markup(),
            parse_mode=ParseMode.HTML
        )
    except Exception as error:
        logger.error(f"Failed to send settings message: {error}")
        await _send_temporary_message(message, StringEnum.SETTINGS_ERROR)


@router.message(Command("settings"))
async def handle_settings_command(message: Message, bot_storage: BotSettings) -> None:
    """Handle /settings command to show bot configuration interface"""
    await _handle_settings_show(message, bot_storage)


async def _update_settings_keyboard(
        callback: CallbackQuery,
        bot_storage: BotSettings,
        chat_id: int,
        notification: str
) -> None:
    """Update the settings keyboard and show notification"""
    try:
        keyboard = SettingsKeyboardBuilder.create(callback.message, bot_storage, chat_id)
        await callback.message.edit_reply_markup(reply_markup=keyboard.as_markup())
        await callback.answer(notification)
    except TelegramBadRequest as e:
        logger.error(f"Failed to update keyboard: {e}")
        await callback.answer(StringEnum.SETTINGS_ERROR)


async def _handle_confirmation(callback: CallbackQuery) -> None:
    """Handle settings confirmation by deleting the message"""
    try:
        await callback.message.delete()
        await callback.answer(StringEnum.SETTINGS_SAVED)
    except TelegramBadRequest as e:
        logger.warning(f"Couldn't delete settings message: {e}")
        await callback.answer(StringEnum.SETTINGS_SAVED_WITH_DELETE_ERROR)


# noinspection PyUnusedLocal
async def _handle_toggle_setting(
        callback: CallbackQuery,
        bot_storage: BotSettings,
        chat_id: int,
        setting_name: str,
        getter: str,
        setter: str,
        template: Literal['BOT_STATUS_UPDATED', 'ERROR_DISPLAY_UPDATED', 'CURRENT_PROBABILITY', 'PROBABILITY_SET'],
        status_map: dict[bool, str]
) -> None:
    """Generic handler for toggle settings"""
    current_value = getattr(bot_storage, getter)(chat_id)
    new_value = not current_value
    getattr(bot_storage, setter)(chat_id, new_value)

    await _update_settings_keyboard(
        callback,
        bot_storage,
        chat_id,
        StringEnum.compose(template, status=status_map[new_value])
    )


async def _handle_probability_adjustment(
        callback: CallbackQuery,
        storage: BotSettings,
        current_prob: int,
        action: SettingsAction,
        chat_id: int
) -> None:
    """Handle probability adjustment with bounds checking"""
    if callback.message.chat.type == ChatType.PRIVATE:
        await callback.answer(StringEnum.PRIVATE_CHAT_UNAVAILABLE)
        return

    step = PROBABILITY_STEP if action == SettingsAction.INCREASE else -PROBABILITY_STEP
    new_prob = SettingsKeyboardBuilder.validate_probability(current_prob + step)

    if new_prob == current_prob:
        await callback.answer(StringEnum.MAX_LIMIT_REACHED)
        return

    storage.set_intervention_probability(chat_id, new_prob)
    await _update_settings_keyboard(
        callback,
        storage,
        chat_id,
        StringEnum.compose('PROBABILITY_SET', prob=str(new_prob))
    )


@router.callback_query(F.data.startswith("settings:"))
async def handle_settings_callback(callback: CallbackQuery, bot_storage: BotSettings) -> None:
    """Process settings keyboard interactions with proper error handling"""
    if not callback.message:
        return

    chat_id = callback.message.chat.id

    if callback.message.chat.type != ChatType.PRIVATE:
        if not await _is_user_admin(callback.bot, chat_id, callback.from_user.id):
            try:
                await callback.answer(StringEnum.ADMIN_ONLY_CALLBACK, show_alert=True)
            except TelegramBadRequest:
                pass
            return

    try:
        _, action_str = callback.data.split(":")

        try:
            action = SettingsAction(action_str)
        except ValueError:
            logger.warning(f"Invalid settings action received: {action_str}")
            await callback.answer(StringEnum.UNSUPPORTED_OPERATION)
            return

        current_prob = bot_storage.get_intervention_probability(chat_id)

        action_handlers = {
            SettingsAction.CONFIRM: partial(_handle_confirmation),
            SettingsAction.TOGGLE_CHAT_ACTIONS: partial(
                _handle_toggle_setting,
                bot_storage=bot_storage,
                chat_id=chat_id,
                setting_name="chat_actions",
                getter="get_chat_actions",
                setter="set_chat_actions",
                template="BOT_STATUS_UPDATED",
                status_map={True: StringEnum.VISIBLE, False: StringEnum.HIDDEN}
            ),
            SettingsAction.TOGGLE_ERRORS: partial(
                _handle_toggle_setting,
                bot_storage=bot_storage,
                chat_id=chat_id,
                setting_name="display_errors",
                getter="get_display_errors",
                setter="set_display_errors",
                template="ERROR_DISPLAY_UPDATED",
                status_map={True: StringEnum.ENABLED, False: StringEnum.DISABLED}
            ),
            SettingsAction.DECREASE: partial(
                _handle_probability_adjustment,
                storage=bot_storage,
                current_prob=current_prob,
                action=SettingsAction.DECREASE,
                chat_id=chat_id
            ),
            SettingsAction.INCREASE: partial(
                _handle_probability_adjustment,
                storage=bot_storage,
                current_prob=current_prob,
                action=SettingsAction.INCREASE,
                chat_id=chat_id
            ),
            SettingsAction.NOOP: partial(
                callback.answer,
                StringEnum.compose('CURRENT_PROBABILITY', prob=str(current_prob))
            )
        }

        handler = action_handlers.get(action)
        if handler:
            await handler(callback=callback)
        else:
            await callback.answer(StringEnum.UNSUPPORTED_OPERATION)

    except Exception as error:
        logger.error(f"Unexpected error in settings callback: {error}", exc_info=True)
        await callback.answer(StringEnum.UNKNOWN_ERROR, show_alert=True)
