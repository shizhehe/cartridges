

## Getting started

1. Install the dependencies:
```bash
cd viz
pip install -r requirements.txt
npm install
```

2. Start the server:
```bash
uvicorn src.server:app --host 0.0.0.0 --port 8001 --reload 
```
If your data is on a remote machine, you need to start the server on the remote, forward the port to your local machine, and run the frontend on your local machine due to CORS issues.

You may need to pick a different port if the default 8001 is already in use.

3. Start the frontend:
Make sure to set the `VITE_API_TARGET` to the port of the server.

```bash
VITE_API_TARGET=http://localhost:8001 npm run dev
```








