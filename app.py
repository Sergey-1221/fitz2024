import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
import re

# Загрузка необходимых данных для nltk
nltk.download('punkt')

# Заголовок приложения
st.title('Анализ текста с использованием SentenceTransformer')

# Текст для обработки (можно заменить на свой)
default_text = """
Машинное обучение продолжает преобразовывать подходы к обработке текстов, предоставляя инструменты для анализа больших объемов данных. Современные алгоритмы, такие как трансформеры, позволяют выявлять глубокие смысловые связи в тексте, делая возможным его использование в различных приложениях.

Технологии обработки естественного языка активно применяются в чат-ботах, системах автоматического перевода и анализе тональности. Например, нейронные сети способны находить скрытые закономерности в текстах, улучшая качество взаимодействия между человеком и машиной.

Тем не менее, создание эффективных моделей требует значительных ресурсов. Обучение глубоких нейронных сетей связано с обработкой больших объемов данных и необходимостью использования мощных вычислительных кластеров для достижения высокой производительности.

Одной из ключевых задач в области текстовой аналитики остается интерпретируемость моделей. Использование внимания в трансформерах не только улучшает точность анализа, но и позволяет понять, какие части текста играют наиболее важную роль при принятии решений.

Интеграция методов обработки текста с другими областями, такими как компьютерное зрение и анализ аудио, открывает путь к созданию мультимодальных систем. Такие подходы используются для генерации контента, автоматической модерации и других задач, требующих обработки сложных данных.

Будущее обработки естественного языка обещает еще более глубокую интеграцию ИИ в повседневную жизнь. Системы станут не просто инструментами, а полноценными интеллектуальными помощниками, способными воспринимать и обрабатывать информацию так же эффективно, как это делает человек.
"""

# Поле для ввода текста
text = st.text_area('Введите текст для обработки', default_text, height=300)

# Параметры
context_char_limit = st.number_input('Максимальное количество символов в одном контексте', min_value=100,
                                     max_value=10000, value=500, step=100)
similarity_threshold = st.slider('Порог сходства для создания ребра', min_value=0.0, max_value=1.0, value=0.5,
                                 step=0.05)


# Кэширование модели для ускорения
@st.cache_resource
def load_model():
    return  ("ai-forever/ru-en-RoSBERTa")


# Кнопка для запуска обработки
if st.button('Обработать текст'):
    with st.spinner('Обработка...'):
        # Шаг 1: Разбиение текста на абзацы
        paragraphs = re.split(r'\n\s*\n', text.strip())

        # Шаг 2: Разбиение абзацев на предложения и получение позиций предложений в тексте
        sentences = []
        sentence_paragraph_mapping = []
        sentence_positions = []

        current_pos = 0  # Текущая позиция в тексте

        for para_idx, paragraph in enumerate(paragraphs):
            para_sentences = nltk.sent_tokenize(paragraph, language='russian')
            for sent in para_sentences:
                # Найти позицию предложения в тексте
                start_idx = text.find(sent, current_pos)
                end_idx = start_idx + len(sent)
                sentences.append(sent)
                sentence_paragraph_mapping.append(para_idx)
                sentence_positions.append((start_idx, end_idx))
                current_pos = end_idx  # Обновить текущую позицию

        # Шаг 3: Вычисление эмбеддингов предложений
        model = load_model()
        sentence_embeddings = model.encode(sentences)

        # Шаг 4: Вычисление матрицы семантического сходства
        similarity_matrix = cosine_similarity(sentence_embeddings)

        # Шаг 5: Построение графа связности с учетом ограничения по абзацам
        G = nx.Graph()

        # Добавление узлов в граф
        for idx, sentence in enumerate(sentences):
            paragraph_idx = sentence_paragraph_mapping[idx]
            G.add_node(idx, sentence=sentence, paragraph=paragraph_idx)

        # Добавление ребер с весами в граф с учетом ближайших абзацев
        num_sentences = len(sentences)
        for i in range(num_sentences):
            para_i = sentence_paragraph_mapping[i]
            for j in range(i + 1, num_sentences):
                para_j = sentence_paragraph_mapping[j]
                # Проверяем, находятся ли предложения в том же или соседних абзацах
                if abs(para_i - para_j) <= 1:
                    similarity = similarity_matrix[i][j]
                    # Устанавливаем порог сходства для создания ребра
                    if similarity > similarity_threshold:
                        G.add_edge(i, j, weight=similarity)

        # Шаг 6: Получение связных компонент графа (контекстов)
        contexts = []
        for component in nx.connected_components(G):
            contexts.append(component)

        # Шаг 7: Ограничение контекстов по количеству символов
        final_contexts = []
        for component in contexts:
            # Получаем список предложений в компоненте
            component_sentences = [sentences[idx] for idx in component]
            # Вычисляем общий размер компоненты в символах
            total_chars = sum(len(sent) for sent in component_sentences)
            if total_chars <= context_char_limit:
                final_contexts.append((component, component_sentences))
            else:
                # Если компонент слишком большой, разбиваем его на меньшие контексты
                sorted_indices = sorted(component, key=lambda x: sentence_positions[x][0])
                temp_context = []
                temp_indices = []
                temp_chars = 0
                for idx in sorted_indices:
                    sent = sentences[idx]
                    sent_length = len(sent)
                    if temp_chars + sent_length <= context_char_limit:
                        temp_context.append(sent)
                        temp_indices.append(idx)
                        temp_chars += sent_length
                    else:
                        if temp_context:
                            final_contexts.append((set(temp_indices), temp_context))
                        temp_context = [sent]
                        temp_indices = [idx]
                        temp_chars = sent_length
                if temp_context:
                    final_contexts.append((set(temp_indices), temp_context))

        # Отображение результатов
        st.subheader('Результаты:')
        for idx, (indices, context_sentences) in enumerate(final_contexts):
            st.markdown(f'**Контекст {idx + 1}:**')
            st.write(' '.join(context_sentences))
            st.write('---')
    st.success('Обработка завершена!')
