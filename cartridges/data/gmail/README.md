
### Setup 
Follow the steps here to get a credentials.json file: https://thepythoncode.com/article/use-gmail-api-in-python
Then move it to cartridges/secrets/credentials.json


### Downloading emails
Go into `cartridges/data/gmail/download.py` and edit the bottom to include the labels you want to download:
```python
if __name__ == "__main__":
    DownloadGmailConfig(
        labels=[
            LabelConfig(name="categories--stanford--primary--stanford-"),
        ],
    ).run()
```

Then run:
```bash
python cartridges/data/gmail/download.py
```



