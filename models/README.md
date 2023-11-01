In this folder the weights of the trained model are to be stored after you download with:
```py
    url = 'https://github.com/system205/PMLDL_Assignment1/releases/download/final-solution/trained_model.zip'

    response = requests.get(url, timeout=100000)
    if response.status_code == 200:
        with open(f"{root}data/external/weights.zip", "wb") as f:
            f.write(response.content)
    else:
        print("Failed to download the zip file.")

    with zipfile.ZipFile(f"{root}data/external/weights.zip", "r") as zip_ref:
        zip_ref.extractall(f'{root}models/')

    os.remove(f"{root}data/external/weights.zip")
```