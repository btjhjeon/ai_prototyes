from googletrans import Translator


def translate(text, src='en', dest='ko'):
    translator = Translator(service_urls=[
        'translate.google.com',
        'translate.google.co.kr',
    ])
    translated = translator.translate(text, src=src, dest=dest)
    if isinstance(translated, (list, tuple)):
        return [t.text for t in translated]
    else:
        return translated.text
