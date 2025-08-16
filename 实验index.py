from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    #   从PDF文件中（按指定页面）提取文字
    paragranphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果制定了页码范围，跳过范围外的也
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' ' + text) if not text.endswith('-') else text.split('-')
        elif buffer:
            paragranphs.append(buffer)
            buffer = ''
    if buffer:
        paragranphs.append(buffer)
    return paragranphs


paragraphs = extract_text_from_pdf("公务接待管理办法.pdf", min_line_length=10)

for para in paragraphs[:4]:
    print(para + "\n")



