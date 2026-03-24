import re
from pathlib import Path


def clean_text(text: str) -> str:
    # clean text from artifacts
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'[^\w\s.,;:!?()\nа-яА-ЯёЁ-]', ' ', text)
    return text.strip()


def classify_document_type(file_path: str, content_preview: str = "") -> str:
    # identifies the document type by name and content
    fname = Path(file_path).name.lower()
    parent_folder = Path(file_path).parent.name.lower()
    
    # pre-release types
    type_mapping = {
        'тз': 'ТЗ', 'техническое задание': 'ТЗ', 'spec': 'ТЗ',
        'расчет': 'Расчет', 'смета': 'Расчет', 'calc': 'Расчет',
        'соглашение': 'Соглашение', 'договор': 'Соглашение', 'contract': 'Соглашение',
        'отчет': 'Отчет', 'report': 'Отчет',
        'приказ': 'Приказ', 'распоряжение': 'Приказ',
        'протокол': 'Протокол', 'protocol': 'Протокол',
        'инструкция': 'Инструкция', 'manual': 'Инструкция',
        'акт': 'Акт',
    }
    
    # search from header
    for key, doc_type in type_mapping.items():
        if key in parent_folder or key in fname:
            return doc_type

    #search from context        
    if content_preview:
        header = content_preview[:600].upper()
        if 'ТЕХНИЧЕСКОЕ ЗАДАНИЕ' in header: return 'ТЗ'
        if 'ДОГОВОР' in header or 'СОГЛАШЕНИЕ' in header: return 'Соглашение'
        if 'ПРИКАЗ' in header: return 'Приказ'
        if 'ОТЧЕТ' in header: return 'Отчет'
        if 'ПРОТОКОЛ' in header: return 'Протокол'
        if 'АКТ' in header: return 'Акт'
        if 'СМЕТА' in header or 'РАСЧЕТ' in header: return 'Расчет'
    
    return "Прочее"


def extract_header(text: str, chunk_text: str) -> str:
    # finds the title of the section closest to the time slot
    header_patterns = [
        r'^(?:РАЗДЕЛ\s+)?[IVX]+\s+([А-ЯЁ][а-яё\s\w-]+?)',
        r'^(?:Глава|Пункт|Раздел)\s*[\d.]+\s*[:.\s]*([А-ЯЁ][^\n]+)',
        r'^([А-ЯЁ]{2,}[\s\w-]+?(?:РАБОТ|ДОГОВОР|СПОРОВ|ЦЕНА|СТОИМОСТЬ))',
    ]
    
    # slide window
    search_window = 1500
    pos = text.find(chunk_text[:50])
    if pos < 0:
        pos = 0
    before = text[max(0, pos - search_window):pos]
    
    lines = before.split('\n')[::-1]
    for line in lines:
        line_stripped = line.strip()
        if len(line_stripped) < 5 or len(line_stripped) > 150:
            continue
        for pattern in header_patterns:
            match = re.search(pattern, line_stripped, re.IGNORECASE)
            if match:
                return match.group(1).strip() if match.lastindex else line_stripped
    
    return "Общий раздел"


def is_structural_chunk(text: str) -> bool:
    # remove the rubbish
    patterns = [
        r'^СОГЛАСОВАНО', r'^УТВЕРЖДАЮ', r'^ЗАТВЕРДЖУЮ',
        r'^Приложение\s+[А-Я]', r'^Лист\s+регистрации',
        r'^Содержание\s*$', r'^Оглавление\s*$',
        r'^ТЕРМИНЫ\s*И\s*ОПРЕДЕЛЕНИЯ',
    ]
    return any(re.match(p, text.strip(), re.IGNORECASE) for p in patterns)