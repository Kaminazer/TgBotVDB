import json
from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ConversationHandler, CallbackContext, filters
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Стани для ConversationHandler
TYPING_DB_NAME = 1
TYPING_COLLECTION_NAME = 2
TYPING_VECTOR_NAME = 3

# Зчитуємо конфігураційний файл
with open('config.json', 'r') as f:
    config = json.load(f)

# Ваш токен, отриманий від BotFather
TOKEN = config['telegram_token']

# Глобальні змінні для підключення до баз даних
milvus_connection = None
qdrant_client = None

# Функція для отримання списку доступних баз даних з конфігурації
def get_available_dbs():
    return ['milvus', 'qdrant']

# Функція, яка обробляє команду /start
async def start(update: Update, context: CallbackContext) -> None:
    global active_db
    await update.message.reply_text(f'Привіт! Поточна активна база даних: {active_db}')

# Функція для надання допомоги /help
async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        'Доступні команди:\n'
        '/use_db - Змінити активну базу даних\n'
        '/create_collection - Створити нову колекцію в активній базі даних\n'
        '/insert_vectors - Додати вектори до колекції в активній базі даних\n'
        '/cancel - Скасувати поточну операцію\n'
    )

# Глобальна змінна для поточної активної бази даних
active_db = 'milvus'

# Функція для зміни активної бази даних
async def use_db(update: Update, context: CallbackContext) -> int:
    available_dbs = get_available_dbs()
    db_list = '\n'.join(available_dbs)
    await update.message.reply_text(f'Доступні бази даних:\n{db_list}\nВведіть назву бази даних, яку хочете використовувати:')
    return TYPING_DB_NAME

async def use_db_end(update: Update, context: CallbackContext) -> int:
    global active_db, milvus_connection, qdrant_client
    active_db = update.message.text
    if active_db in get_available_dbs():
        # Підключення до обраної бази даних
        if active_db == 'milvus':
            milvus_connection = connections.connect("default", host=config['milvus']['host'], port=config['milvus']['port'])
        elif active_db == 'qdrant':
            qdrant_client = QdrantClient(host=config['qdrant']['host'], port=config['qdrant']['port'])
        await update.message.reply_text(f'Активна база даних: {active_db}')
    else:
        await update.message.reply_text(f'Невідома база даних: {active_db}')
    return ConversationHandler.END

# Функція для початку створення колекції
async def create_collection_start(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text('Введіть назву колекції для створення:')
    return TYPING_COLLECTION_NAME

# Функція для завершення створення колекції
async def create_collection_end(update: Update, context: CallbackContext) -> int:
    global active_db, qdrant_client
    collection_name = update.message.text
    try:
        if active_db == 'milvus':
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
            ]
            schema = CollectionSchema(fields)
            collection = Collection(name=collection_name, schema=schema)
            await update.message.reply_text(f'Колекцію {collection_name} успішно створено в {active_db}.')
        elif active_db == 'qdrant':
            # Перевірка, чи qdrant_client створений
            if qdrant_client is None:
                qdrant_client = QdrantClient(host=config['qdrant']['host'], port=config['qdrant']['port'])
            
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE)
            )
            await update.message.reply_text(f'Колекцію {collection_name} успішно створено в {active_db}.')
        else:
            await update.message.reply_text(f'Помилка: невідома база даних {active_db}.')
    except Exception as e:
        await update.message.reply_text(f'Помилка при створенні колекції: {str(e)}')
    return ConversationHandler.END

# Функція для початку додавання векторів до колекції
async def insert_vectors_start(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text('Введіть назву колекції для додавання векторів:')
    return TYPING_VECTOR_NAME

# Функція для завершення додавання векторів до колекції
async def insert_vectors_end(update: Update, context: CallbackContext) -> int:
    global active_db, qdrant_client
    collection_name = update.message.text
    try:
        if active_db == 'milvus':
            collection = Collection(name=collection_name)
            vector_dim = collection.schema[1].params['dim']
            embeddings = np.random.random((1000, vector_dim)).tolist()
            collection.insert([embeddings])
            await update.message.reply_text(f'Вектори додано до колекції {collection_name} в {active_db}.')
        elif active_db == 'qdrant':
            # Перевірка, чи qdrant_client створений
            if qdrant_client is None:
                qdrant_client = QdrantClient(host=config['qdrant']['host'], port=config['qdrant']['port'])
            
            collection_info = qdrant_client.get_collection(collection_name)
            vector_dim = collection_info.vectors_params.size
            vectors = np.random.random((1000, vector_dim)).tolist()
            points = [
                models.PointStruct(
                    id=idx, 
                    vector=vector
                ) for idx, vector in enumerate(vectors)
            ]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            await update.message.reply_text(f'Вектори додано до колекції {collection_name} в {active_db}.')
        else:
            await update.message.reply_text(f'Помилка: невідома база даних {active_db}.')
    except Exception as e:
        await update.message.reply_text(f'Помилка при додаванні векторів: {str(e)}')
    return ConversationHandler.END

# Функція для припинення розмови
async def cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text('Операцію скасовано.')
    return ConversationHandler.END

def main() -> None:
    # Створюємо Application
    application = ApplicationBuilder().token(TOKEN).build()

    # Реєструємо команду /start
    application.add_handler(CommandHandler("start", start))

    # Реєструємо команду /help
    application.add_handler(CommandHandler("help", help_command))

    # Реєструємо команду для зміни активної бази даних
    use_db_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('use_db', use_db)],
        states={
            TYPING_DB_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, use_db_end)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    application.add_handler(use_db_conv_handler)

    # Реєструємо розмовний режим для створення колекції та додавання векторів
    col_vec_conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('create_collection', create_collection_start),
            CommandHandler('insert_vectors', insert_vectors_start)
        ],
        states={
            TYPING_COLLECTION_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, create_collection_end)
            ],
            TYPING_VECTOR_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, insert_vectors_end)
            ]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    application.add_handler(col_vec_conv_handler)

    # Запускаємо бота
    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()