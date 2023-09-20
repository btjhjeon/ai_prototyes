from huggingface_hub import login


def login_huggingface(token=None):
    if token:
        login(token=token)
    else:
        login()
