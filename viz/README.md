

## Starting the server
If your data is on a remote machine, you need to start the server and frontend on different machines. 

First run the following on the remote machine in Cursor/VSCode so that the port is automatically forwarded to your local machine:
```bash
uvicorn viz.src.server:app --host 0.0.0.0 --port 8001 --reload
```

Then run the following on your local machine:
```bash
cd viz
VITE_API_TARGET=http://localhost:8001 npm run dev
```
Make sure the `VITE_API_TARGET` is set to forwarded port. You can check this in VSCode by going to "Ports: Focus on Ports View" from the command palette (cmd+shift+p).







